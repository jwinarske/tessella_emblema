//! Turning a resolved override back into a specialization constant.
//!
//! `tests/naga_overrides.rs` establishes the two facts this module exists between. naga's SPIR-V
//! backend refuses a module that still carries a WGSL `override`, so the constant cannot be
//! emitted as a specialization constant directly; and `process_overrides` substitutes a value
//! without folding the branch, so the emitted module still carries a boolean `OpConstant` and an
//! `OpBranchConditional` reading it.
//!
//! What is left is a rewrite. Find that constant, change its opcode to the specialization form,
//! and decorate it with a `SpecId`. The driver then folds the branch and its dead arm at
//! `vkCreateGraphicsPipelines`, which is the point: on the gating target a fragment shader costs
//! what its whole body needs rather than what the taken branch needs, so an unused arm has to be
//! gone before the pipeline is compiled and not merely skipped at run time.
//!
//! The result is one SPIR-V module per (family, surface), with permutations as arguments to
//! pipeline creation — no module per permutation, and nothing translated at run time.
//!
//! # Finding the constant
//!
//! By name. naga emits an `OpName` for a substituted override carrying the name it had in the
//! WGSL, so `override color_from_attribute: bool` becomes an `OpName` of
//! `"color_from_attribute"` against the id of an `OpConstantFalse`. A name that is absent, or
//! names something that is not a boolean constant, is an error — never a silent no-op, because a
//! permutation switch that quietly stopped being one would cost a shader its folding and show up
//! only as a frame time.

use crate::spirv;

/// A permutation switch to specialize: the `override`'s name, and the `SpecId` to bind it to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Switch<'a> {
    /// The name the constant had in the WGSL, which naga wrote into an `OpName`.
    pub name: &'a str,
    /// The `SpecId` the pipeline will set this by. Unique within a module; the caller owns the
    /// numbering, because it is the same number the pipeline's `VkSpecializationMapEntry` uses.
    pub id: u32,
}

/// Why a module could not be specialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The words are not a SPIR-V module, or are truncated mid-instruction.
    NotSpirv,
    /// No `OpName` in the module carries this name.
    ///
    /// Carries the name, since the fix is to reconcile it with the WGSL rather than to retry.
    NoSuchName(String),
    /// The name resolves to an id that is not an `OpConstantTrue` or `OpConstantFalse`.
    ///
    /// Either the override was not a `bool`, or naga folded it after all — the second would mean
    /// this whole module has stopped being necessary, which `tests/naga_overrides.rs` is what
    /// notices.
    NotABooleanConstant(String),
    /// Two switches asked for the same `SpecId`.
    DuplicateSpecId(u32),
}

/// Rewrites each named boolean constant into a specialization constant bound to its `SpecId`.
///
/// The opcodes are the same length, so the constants are patched where they lie; the `SpecId`
/// decorations are inserted at the head of the annotations block, which is where the physical
/// layout puts them — after the debug names and before the types those constants belong to.
///
/// The module's id bound is unchanged: nothing new is declared, only decorated.
///
/// # Errors
///
/// [`Error`], and the module is left untouched when one is returned: the words are only written
/// after every switch has been resolved.
pub fn specialize(words: &[u32], switches: &[Switch<'_>]) -> Result<Vec<u32>, Error> {
    for (at, switch) in switches.iter().enumerate() {
        if switches[..at].iter().any(|prior| prior.id == switch.id) {
            return Err(Error::DuplicateSpecId(switch.id));
        }
    }

    // Resolve every name first. A module half-specialized because the third switch was misspelled
    // would be a shader that compiles, draws, and is slow for a reason nothing reports.
    let mut targets = Vec::with_capacity(switches.len());
    for switch in switches {
        let id = spirv::id_named(words, switch.name)
            .ok_or_else(|| Error::NoSuchName(switch.name.into()))?;
        let at = spirv::boolean_constant_at(words, id)
            .ok_or_else(|| Error::NotABooleanConstant(switch.name.into()))?;
        targets.push((at, switch.id));
    }

    let mut out = words.to_vec();
    for (at, _) in &targets {
        let opcode = (out[*at] & 0xFFFF) as u16;
        let specialized = match opcode {
            spirv::OP_CONSTANT_TRUE => spirv::OP_SPEC_CONSTANT_TRUE,
            spirv::OP_CONSTANT_FALSE => spirv::OP_SPEC_CONSTANT_FALSE,
            _ => return Err(Error::NotSpirv),
        };
        // Word count is unchanged -- both forms are three words -- so only the opcode moves.
        out[*at] = (out[*at] & 0xFFFF_0000) | u32::from(specialized);
    }

    let mut decorations = Vec::with_capacity(targets.len() * 4);
    for (at, spec_id) in &targets {
        let id = out[*at + 2];
        // OpDecorate <target> SpecId <n>: four words, the count in the high half.
        decorations.extend_from_slice(&[
            (4 << 16) | u32::from(spirv::OP_DECORATE),
            id,
            spirv::DECORATION_SPEC_ID,
            *spec_id,
        ]);
    }
    let insert = spirv::annotations_begin(words).ok_or(Error::NotSpirv)?;
    out.splice(insert..insert, decorations);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Error, Switch, specialize};
    use crate::spirv;
    use std::collections::HashMap;

    /// Two permutation switches over one family's shape, as a real module has.
    const SHADER: &str = r"
override color_from_attribute: bool = false;
override has_halo: bool = true;

@fragment
fn main(@location(0) attribute_color: vec4<f32>) -> @location(0) vec4<f32> {
    var out = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    if color_from_attribute { out = attribute_color; }
    if has_halo { out = out * 0.5; }
    return out;
}
";

    /// The module as the build would hand it over: overrides substituted, nothing specialized.
    fn emitted() -> Vec<u32> {
        let module = naga::front::wgsl::parse_str(SHADER).expect("parses");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
        let mut values: HashMap<String, f64> = HashMap::new();
        values.insert("color_from_attribute".into(), 0.0);
        values.insert("has_halo".into(), 1.0);
        let (resolved, resolved_info) =
            naga::back::pipeline_constants::process_overrides(&module, &info, &values)
                .expect("substitutes");
        naga::back::spv::write_vec(
            &resolved,
            &resolved_info,
            &naga::back::spv::Options::default(),
            None,
        )
        .expect("emits")
    }

    /// Every instruction's opcode and operands, for asserting on the result.
    fn instructions(words: &[u32]) -> Vec<(u16, Vec<u32>)> {
        let mut out = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let opcode = (words[at] & 0xFFFF) as u16;
            let count = (words[at] >> 16) as usize;
            assert!(count > 0 && at + count <= words.len(), "truncated at {at}");
            out.push((opcode, words[at + 1..at + count].to_vec()));
            at += count;
        }
        out
    }

    /// Both switches become specialization constants, each decorated with the id it was given.
    #[test]
    fn a_switch_becomes_a_specialization_constant() {
        let words = emitted();
        let out = specialize(
            &words,
            &[
                Switch {
                    name: "color_from_attribute",
                    id: 0,
                },
                Switch {
                    name: "has_halo",
                    id: 1,
                },
            ],
        )
        .expect("specializes");

        let after = instructions(&out);
        let specialized: Vec<u32> = after
            .iter()
            .filter(|(op, _)| {
                *op == spirv::OP_SPEC_CONSTANT_TRUE || *op == spirv::OP_SPEC_CONSTANT_FALSE
            })
            .map(|(_, operands)| operands[1])
            .collect();
        assert_eq!(specialized.len(), 2, "both switches specialized");

        let decorated: Vec<(u32, u32)> = after
            .iter()
            .filter(|(op, operands)| {
                *op == spirv::OP_DECORATE && operands.get(1) == Some(&spirv::DECORATION_SPEC_ID)
            })
            .map(|(_, operands)| (operands[0], operands[2]))
            .collect();
        assert_eq!(decorated.len(), 2, "both carry a SpecId");
        for (id, _) in &decorated {
            assert!(
                specialized.contains(id),
                "a SpecId decorates a spec constant"
            );
        }

        // The values survive: false stays false until a pipeline says otherwise.
        assert_eq!(
            after
                .iter()
                .filter(|(op, _)| *op == spirv::OP_SPEC_CONSTANT_FALSE)
                .count(),
            1
        );
        assert_eq!(
            after
                .iter()
                .filter(|(op, _)| *op == spirv::OP_SPEC_CONSTANT_TRUE)
                .count(),
            1
        );
    }

    /// The branch that reads the constant is still there, which is what the driver will fold.
    ///
    /// A rewrite that specialized the constant and lost the branch would produce a shader whose
    /// permutation does nothing -- one arm, always, whatever the pipeline sets.
    #[test]
    fn the_branches_survive_the_rewrite() {
        let words = emitted();
        let before = instructions(&words)
            .iter()
            .filter(|(op, _)| *op == spirv::OP_BRANCH_CONDITIONAL)
            .count();
        let out = specialize(
            &words,
            &[
                Switch {
                    name: "color_from_attribute",
                    id: 0,
                },
                Switch {
                    name: "has_halo",
                    id: 1,
                },
            ],
        )
        .expect("specializes");
        let after = instructions(&out)
            .iter()
            .filter(|(op, _)| *op == spirv::OP_BRANCH_CONDITIONAL)
            .count();
        assert_eq!(
            before, after,
            "the rewrite touches constants, not control flow"
        );
        assert!(before > 0, "there were branches to keep");
    }

    /// The decorations land in the annotations block, before the types and constants.
    ///
    /// Not cosmetic: a decoration after the declaration it decorates is a layout violation, and
    /// the one place it would be noticed is a driver that rejects the module on a board.
    #[test]
    fn the_decoration_precedes_the_constant_it_decorates() {
        let words = emitted();
        let out = specialize(
            &words,
            &[Switch {
                name: "has_halo",
                id: 7,
            }],
        )
        .expect("specializes");
        let after = instructions(&out);
        let decorate = after
            .iter()
            .position(|(op, operands)| {
                *op == spirv::OP_DECORATE && operands.get(1) == Some(&spirv::DECORATION_SPEC_ID)
            })
            .expect("a SpecId decoration");
        let constant = after
            .iter()
            .position(|(op, _)| *op == spirv::OP_SPEC_CONSTANT_TRUE)
            .expect("the specialized constant");
        assert!(decorate < constant, "annotations precede declarations");
    }

    /// The id bound is untouched, because nothing new is declared.
    #[test]
    fn the_id_bound_does_not_move() {
        let words = emitted();
        let out = specialize(
            &words,
            &[Switch {
                name: "has_halo",
                id: 0,
            }],
        )
        .expect("specializes");
        assert_eq!(out[3], words[3], "same bound: only decorations were added");
    }

    /// A name the module does not have is an error, and says which.
    #[test]
    fn an_unknown_switch_is_refused() {
        let words = emitted();
        let out = specialize(
            &words,
            &[Switch {
                name: "no_such_switch",
                id: 0,
            }],
        );
        assert_eq!(out, Err(Error::NoSuchName("no_such_switch".into())));
    }

    /// And a name that is not a boolean constant is refused too, rather than patched blind.
    #[test]
    fn a_name_that_is_not_a_boolean_constant_is_refused() {
        let words = emitted();
        // `main` is named in the module, and is a function.
        let out = specialize(
            &words,
            &[Switch {
                name: "main",
                id: 0,
            }],
        );
        assert_eq!(out, Err(Error::NotABooleanConstant("main".into())));
    }

    /// A batch with one bad switch yields nothing, rather than a module missing one switch.
    ///
    /// Weak as a guard and kept knowingly: taking `&[u32]` and returning a fresh `Vec` makes a
    /// partial rewrite unrepresentable, so this passes by construction rather than by vigilance.
    /// It is here to fail the day someone refactors to rewrite in place, which is when the
    /// property stops being free -- a module specialized up to the misspelled switch would
    /// compile, draw, and be slow for a reason nothing reports.
    #[test]
    fn one_bad_switch_leaves_the_module_alone() {
        let words = emitted();
        let out = specialize(
            &words,
            &[
                Switch {
                    name: "has_halo",
                    id: 0,
                },
                Switch {
                    name: "no_such_switch",
                    id: 1,
                },
            ],
        );
        assert!(out.is_err(), "the batch is refused");
        // And the original is unchanged, being borrowed rather than written through.
        assert_eq!(
            instructions(&words)
                .iter()
                .filter(|(op, _)| *op == spirv::OP_SPEC_CONSTANT_TRUE)
                .count(),
            0
        );
    }

    /// Two switches cannot share a `SpecId`; the pipeline could not tell them apart.
    #[test]
    fn a_duplicate_spec_id_is_refused() {
        let words = emitted();
        let out = specialize(
            &words,
            &[
                Switch {
                    name: "has_halo",
                    id: 3,
                },
                Switch {
                    name: "color_from_attribute",
                    id: 3,
                },
            ],
        );
        assert_eq!(out, Err(Error::DuplicateSpecId(3)));
    }

    /// Specializing nothing returns the module unchanged, rather than a subtly different one.
    #[test]
    fn no_switches_is_the_identity() {
        let words = emitted();
        assert_eq!(specialize(&words, &[]), Ok(words.clone()));
    }

    /// Words that are not a module are refused rather than indexed into.
    #[test]
    fn a_truncated_module_is_refused() {
        let out = specialize(&[0x0723_0203, 1], &[Switch { name: "x", id: 0 }]);
        assert_eq!(out, Err(Error::NoSuchName("x".into())));
        assert_eq!(specialize(&[], &[]), Err(Error::NotSpirv));
    }
}
