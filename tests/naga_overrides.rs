//! What naga does with a WGSL `override`, and what that leaves of the permutation mechanism.
//!
//! A permutation of a shader family — "this
//! layer's color comes from an attribute, that one's is uniform" — has to be *folded out* at
//! pipeline creation, not branched around at run time, because on the gating target a fragment
//! shader costs what its whole body needs and not what the taken branch needs: three unrelated
//! additions once cost about ten per cent on a path that used none of them.
//!
//! There were two possible shapes. Either naga emits `OpSpecConstantTrue`/`False` with a `SpecId`
//! decoration and the mechanism is literal, or it resolves overrides before emission and the
//! fallback is a build-time SPIR-V rewrite: emit each constant as a boolean `OpConstant` carrying
//! a marker and patch it to `OpSpecConstantFalse` plus a `SpecId`.
//!
//! Neither is what happens, and the tests below are the evidence.

use std::collections::HashMap;

/// SPIR-V opcodes, from the core grammar: the low sixteen bits of an instruction's first word.
const OP_DECORATE: u16 = 71;
const OP_CONSTANT_TRUE: u16 = 41;
const OP_CONSTANT_FALSE: u16 = 42;
const OP_SPEC_CONSTANT_TRUE: u16 = 49;
const OP_SPEC_CONSTANT_FALSE: u16 = 50;
const OP_BRANCH_CONDITIONAL: u16 = 250;
/// `Decoration::SpecId`, the operand that follows a decorated id.
const DECORATION_SPEC_ID: u32 = 1;

/// One family's shape in miniature: a boolean permutation switch and a branch on it.
const SHADER: &str = r"
override color_from_attribute: bool = false;

@fragment
fn main(@location(0) attribute_color: vec4<f32>) -> @location(0) vec4<f32> {
    if color_from_attribute {
        return attribute_color;
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
";

/// Counts of the instructions this question turns on.
#[derive(Debug, Default, PartialEq, Eq)]
struct Found {
    spec_constants: usize,
    spec_ids: usize,
    plain_bool_constants: usize,
    conditional_branches: usize,
}

/// Walks a SPIR-V module's instruction stream.
///
/// Hand-rolled rather than pulled from a crate: a five-word header and a word count per
/// instruction is the whole format needed, and this test exists to avoid depending on a parser's
/// opinion of what it found.
fn scan(words: &[u32]) -> Found {
    let mut found = Found::default();
    let mut at = 5;
    while at < words.len() {
        let opcode = (words[at] & 0xFFFF) as u16;
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            break;
        }
        let operands = &words[at + 1..at + count];
        match opcode {
            OP_SPEC_CONSTANT_TRUE | OP_SPEC_CONSTANT_FALSE => found.spec_constants += 1,
            OP_CONSTANT_TRUE | OP_CONSTANT_FALSE => found.plain_bool_constants += 1,
            OP_BRANCH_CONDITIONAL => found.conditional_branches += 1,
            OP_DECORATE if operands.get(1) == Some(&DECORATION_SPEC_ID) => found.spec_ids += 1,
            _ => {}
        }
        at += count;
    }
    found
}

/// Parses and validates, returning the module and its info.
fn front(source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source).expect("the shader parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("the shader validates");
    (module, info)
}

/// **The answer: naga's SPIR-V backend refuses a module that still has an override.**
///
/// Not "resolves it first" — it will not emit at all, returning `spv::Error::Override`. So the
/// literal form is unavailable, and so is any scheme that hoped to catch the override on
/// its way through the backend.
#[test]
fn the_spirv_backend_refuses_an_unresolved_override() {
    let (module, info) = front(SHADER);
    let emitted =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None);
    assert!(
        matches!(emitted, Err(naga::back::spv::Error::Override)),
        "naga now emits SPIR-V for a module with overrides; re-read this question"
    );
}

/// And substituting the value leaves exactly what a build-time rewrite needs.
///
/// This is the half that decides the mechanism, and it is the opposite of the worry.
/// `process_overrides` does **not** constant-fold the branch: the module comes out carrying one
/// boolean `OpConstant` and one `OpBranchConditional` that reads it. So there is a constant to
/// patch and a branch for it to steer.
///
/// The fallback therefore works as written: emit each permutation switch through
/// `process_overrides` with a known value, then rewrite that `OpConstantFalse` to
/// `OpSpecConstantFalse` and decorate it `SpecId`. The driver folds the branch and the dead arm at
/// `vkCreateGraphicsPipelines`, which is the whole point — on V3D a shader costs what its body
/// needs, so the unused arm has to be gone before the pipeline is compiled, not skipped at run
/// time.
///
/// One module per (family, surface), permutations as pipeline-creation arguments, no translator at
/// run time, and no module-count explosion: a couple of hundred lines of tooling, not a shader-cost
/// regression.
#[test]
fn a_substituted_override_leaves_a_constant_and_a_branch_to_patch() {
    let (module, info) = front(SHADER);
    let mut values: HashMap<String, f64> = HashMap::new();
    values.insert("color_from_attribute".to_owned(), 0.0);
    let (resolved, resolved_info) =
        naga::back::pipeline_constants::process_overrides(&module, &info, &values)
            .expect("the override substitutes");

    let words = naga::back::spv::write_vec(
        &resolved,
        &resolved_info,
        &naga::back::spv::Options::default(),
        None,
    )
    .expect("the resolved module emits");

    assert_eq!(
        scan(&words),
        Found {
            // Nothing specialized yet -- that is what the rewrite adds.
            spec_constants: 0,
            spec_ids: 0,
            // The switch, still a constant in the module.
            plain_bool_constants: 1,
            // And the branch reading it, un-folded. If naga ever folds this, the rewrite has
            // nothing to steer and the mechanism needs rethinking, which is why this is pinned.
            conditional_branches: 1,
        }
    );
}

/// The same module with the override set the other way is a *different* module, not a variant.
///
/// Which is the cost: permutations become SPIR-V modules rather than pipeline-creation arguments,
/// and the count is per (family, surface, permutation) instead of per (family, surface).
#[test]
fn each_value_is_its_own_module() {
    let (module, info) = front(SHADER);
    let emit = |value: f64| {
        let mut values: HashMap<String, f64> = HashMap::new();
        values.insert("color_from_attribute".to_owned(), value);
        let (resolved, resolved_info) =
            naga::back::pipeline_constants::process_overrides(&module, &info, &values)
                .expect("the override substitutes");
        naga::back::spv::write_vec(
            &resolved,
            &resolved_info,
            &naga::back::spv::Options::default(),
            None,
        )
        .expect("the resolved module emits")
    };
    assert_ne!(
        emit(0.0),
        emit(1.0),
        "the two permutations differ, so each is its own SPIR-V"
    );
}
