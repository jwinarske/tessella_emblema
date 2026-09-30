//! Just enough SPIR-V to find and decorate a constant.
//!
//! Hand-rolled rather than taken from a crate, and deliberately so: a five-word header and a word
//! count in the top half of each instruction is the whole of the format this crate reads, and a
//! parser that rebuilt the module into its own types would be a second opinion about what naga
//! emitted rather than a reading of it. Nothing here allocates or reorders — it locates.
//!
//! The physical layout is what the section boundaries below rely on: capability and mode
//! declarations, then debug instructions, then annotations, then types, constants and global
//! variables, then functions. A decoration must reach the annotations block; one sitting after the
//! declaration it decorates is invalid, and the place that would be noticed is a driver on a board
//! rejecting the module.

/// The first word of every SPIR-V module.
const MAGIC: u32 = 0x0723_0203;

/// `OpName`, which is how a substituted override is found: naga writes the WGSL name here.
pub(crate) const OP_NAME: u16 = 5;
/// `OpConstantTrue` and `OpConstantFalse`, what an override becomes once substituted.
pub(crate) const OP_CONSTANT_TRUE: u16 = 41;
/// See [`OP_CONSTANT_TRUE`].
pub(crate) const OP_CONSTANT_FALSE: u16 = 42;
/// `OpSpecConstantTrue`, the same declaration a pipeline can override. Three words, as the plain
/// form is, which is what lets the rewrite patch in place.
pub(crate) const OP_SPEC_CONSTANT_TRUE: u16 = 49;
/// See [`OP_SPEC_CONSTANT_TRUE`].
pub(crate) const OP_SPEC_CONSTANT_FALSE: u16 = 50;
/// `OpDecorate`, which carries the `SpecId`.
pub(crate) const OP_DECORATE: u16 = 71;
/// `OpBranchConditional`, the branch the driver folds once the constant is specialized.
///
/// Read only by the tests that check the rewrite leaves control flow alone, which is why it is
/// gated: a rewrite that specialized the constant and lost the branch would give every pipeline
/// the same arm, and that is worth a test but not a constant in the shipped library.
#[cfg(test)]
pub(crate) const OP_BRANCH_CONDITIONAL: u16 = 250;
/// `Decoration::SpecId`.
pub(crate) const DECORATION_SPEC_ID: u32 = 1;

/// Opcodes of the preamble and debug sections, which together precede the annotations.
///
/// `OpCapability`, `OpExtension`, `OpExtInstImport`, `OpMemoryModel`, `OpEntryPoint`,
/// `OpExecutionMode`, `OpExecutionModeId`; then `OpSourceContinued`, `OpSource`,
/// `OpSourceExtension`, `OpName`, `OpMemberName`, `OpString`, `OpModuleProcessed`.
const BEFORE_ANNOTATIONS: [u16; 14] = [17, 10, 11, 14, 15, 16, 331, 2, 3, 4, 5, 6, 7, 330];

/// Walks the instruction stream, yielding `(word index, opcode, operand words)`.
///
/// Stops at the first malformed instruction rather than panicking, so a truncated module reads as
/// a short one and the callers above turn that into a refusal.
fn instructions(words: &[u32]) -> impl Iterator<Item = (usize, u16, &[u32])> {
    let mut at = if words.len() >= 5 && words[0] == MAGIC {
        5
    } else {
        words.len()
    };
    core::iter::from_fn(move || {
        if at >= words.len() {
            return None;
        }
        let opcode = (words[at] & 0xFFFF) as u16;
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            at = words.len();
            return None;
        }
        let here = at;
        at += count;
        Some((here, opcode, &words[here + 1..here + count]))
    })
}

/// The id an `OpName` gives this name, if any.
///
/// The string is the instruction's remaining words as little-endian bytes, NUL-terminated and
/// padded — read here rather than through a helper so the padding rule stays visible: a name whose
/// length is a multiple of four still carries a whole word of zeros after it.
pub(crate) fn id_named(words: &[u32], name: &str) -> Option<u32> {
    instructions(words)
        .filter(|(_, opcode, operands)| *opcode == OP_NAME && !operands.is_empty())
        .find(|(_, _, operands)| {
            let mut bytes = Vec::new();
            'words: for word in &operands[1..] {
                for byte in word.to_le_bytes() {
                    if byte == 0 {
                        break 'words;
                    }
                    bytes.push(byte);
                }
            }
            bytes == name.as_bytes()
        })
        .map(|(_, _, operands)| operands[0])
}

/// Where the boolean constant with this id is declared, as a word index.
///
/// `None` when the id is declared as something else, which the caller reports rather than patches:
/// an override that is not a `bool`, or a module naga folded after all.
pub(crate) fn boolean_constant_at(words: &[u32], id: u32) -> Option<usize> {
    instructions(words)
        .find(|(_, opcode, operands)| {
            (*opcode == OP_CONSTANT_TRUE || *opcode == OP_CONSTANT_FALSE)
                && operands.get(1) == Some(&id)
        })
        .map(|(at, _, _)| at)
}

/// The word index where the annotations block begins.
///
/// That is the first instruction past the preamble and the debug names — which is where existing
/// decorations already sit, and where new ones belong. A module of nothing but a preamble has its
/// annotations block at the end, which is still the right place to insert.
pub(crate) fn annotations_begin(words: &[u32]) -> Option<usize> {
    if words.len() < 5 || words[0] != MAGIC {
        return None;
    }
    Some(
        instructions(words)
            .find(|(_, opcode, _)| !BEFORE_ANNOTATIONS.contains(opcode))
            .map_or(words.len(), |(at, _, _)| at),
    )
}

#[cfg(test)]
mod tests {
    use super::{MAGIC, annotations_begin, boolean_constant_at, id_named};

    /// `OpTypeVoid`, the one type this fixture needs to sit past the debug section.
    const OP_TYPE_VOID: u32 = 19;

    /// A module with a header, one `OpName`, and one `OpTypeVoid` after it.
    fn module() -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0000, 0, 10, 0];
        // OpName %4 "flag": 2 fixed words + 2 for "flag\0\0\0\0" = 4 words.
        words.extend_from_slice(&[(4 << 16) | 5, 4, u32::from_le_bytes(*b"flag"), 0]);
        // OpTypeVoid %5, which is past the debug section.
        words.extend_from_slice(&[(2 << 16) | OP_TYPE_VOID, 5]);
        words
    }

    #[test]
    fn a_name_resolves_to_its_id() {
        assert_eq!(id_named(&module(), "flag"), Some(4));
    }

    /// A prefix is not a match: `fla` must not find `flag`.
    #[test]
    fn a_shorter_name_does_not_match() {
        assert_eq!(id_named(&module(), "fla"), None);
    }

    #[test]
    fn an_absent_name_is_none() {
        assert_eq!(id_named(&module(), "other"), None);
    }

    /// Annotations begin at the first instruction past the debug section.
    #[test]
    fn annotations_begin_after_the_names() {
        // 5 header + 4 for the OpName = 9.
        assert_eq!(annotations_begin(&module()), Some(9));
    }

    /// Something that is not a module is refused rather than indexed into.
    #[test]
    fn a_non_module_has_no_sections() {
        assert_eq!(annotations_begin(&[]), None);
        assert_eq!(annotations_begin(&[1, 2, 3, 4, 5]), None);
        assert_eq!(id_named(&[1, 2, 3, 4, 5], "flag"), None);
    }

    /// A word count of zero would loop forever if it were trusted.
    #[test]
    fn a_zero_word_count_terminates() {
        let words = vec![MAGIC, 0x0001_0000, 0, 10, 0, 0];
        assert_eq!(id_named(&words, "flag"), None);
        assert_eq!(boolean_constant_at(&words, 4), None);
    }

    /// And a count running past the end stops rather than reading off it.
    #[test]
    fn a_truncated_instruction_terminates() {
        let words = vec![MAGIC, 0x0001_0000, 0, 10, 0, (9 << 16) | 5, 4];
        assert_eq!(id_named(&words, "flag"), None);
    }
}
