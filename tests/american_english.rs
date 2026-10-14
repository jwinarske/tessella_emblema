// SPDX-License-Identifier: BSD-2-Clause
//! Everything in this crate a reader sees is written in American English.
//!
//! A stated convention, and one that drifts silently: nothing about "colour" beside "color" fails
//! to compile, and a doc comment is as much a part of the interface as the name it belongs to.
//! `emblema` carries this check and `tessella` now does too, and this crate sits between them --
//! the two trees it reads and the one it is read by all hold to the same English, so a word
//! crossing from one of them should not arrive here unexamined.
//!
//! Three had, all in benches: a neighbouring value, a 3x3 neighbourhood, and a `cancelled` that
//! stays, for the reason [`EXEMPT`] gives.
//!
//! # How the words are chosen
//!
//! By spelling rather than by a dictionary, because the list of words this crate actually uses is
//! short and a dictionary would need a list anyway. The entries are the *families* -- -our, -ise
//! and -isation, -re, the doubled -lled, -ogue, the -ce nouns, and the one-offs -- rather than the
//! words this tree happened to get wrong. A word earns its place by belonging to a family, which
//! is what makes the check hold for prose nobody has written yet; a list grown one correction at a
//! time is green until the next word outside it arrives.
//!
//! The search is unanchored, which is what [`ISE`]'s three endings and the two omissions below are
//! for: "recolors" has to be found as surely as "colors", so an entry is a substring and any
//! American word containing it is a false failure.
//!
//! Two words belong to a family and still cannot be searched for. "programme" is inside
//! **programmer** and **programmed**, and this crate's `unprogrammed` batch counter is how that
//! was found; "kerb" is inside **checkerboard**. A substring search has no way to tell either from
//! the American word, so the words are left out rather than exempted file by file. "grey" has the
//! same shape -- it is inside "greyhound" -- and stays, because the color appears in this tree and
//! the dog does not.
//!
//! # What this does not read
//!
//! The walk starts at this crate's own root, so neither path dependency is read. `tessella` and
//! `tessella-consume` answer for their own English, and a tree this one cannot commit to is not
//! one it should fail for. `tests/bench-baselines` is measurement output rather than prose.

use std::path::{Path, PathBuf};

/// British verbs in -ise, as stems, each searched for with the three endings the British verb takes.
///
/// `-ise` (which covers `-ised` and `-ises` too), `-ising`, and `-isation`. Not the bare stem,
/// because the bare stem is inside the American `-ism` and `-ist` nouns: "criticism", "optimism",
/// "organism", "realism", "modernism" and "apologist" are spelled that way on both sides of the
/// Atlantic. `plan.md` says "criticism", and a bare stem list calls it British.
///
/// "analys" is here for the same reason read from the other end -- it is inside "analysis" -- and
/// its three endings give "analyse" and "analysing" without it.
const ISE: [&str; 31] = [
    "recognis",
    "normalis",
    "organis",
    "serialis",
    "visualis",
    "initialis",
    "prioritis",
    "summaris",
    "quantis",
    "analys",
    "realis",
    "minimis",
    "maximis",
    "specialis",
    "standardis",
    "utilis",
    "apologis",
    "criticis",
    "authoris",
    "synchronis",
    "customis",
    "finalis",
    "generalis",
    "modernis",
    "emphasis",
    "characteris",
    "parameteris",
    "rasteris",
    "digitis",
    "optimis",
    "practis",
];

/// Everything else, searched as written.
///
/// Stems where the doubled consonant or the ending is the whole difference, so that every inflected
/// form is caught by one entry: "travell" finds "travelled" and "travelling", neither of which
/// contains the American spelling. Whole words where the American form is shorter and so cannot
/// contain them -- "judgement", "analogue" and "catalogue" are not inside "judgment", "analog" or
/// "catalog".
const BRITISH: [&str; 61] = [
    // -our
    "colour",
    "neighbour",
    "behaviour",
    "favour",
    "honour",
    "flavour",
    "labour",
    "rumour",
    "vapour",
    "armour",
    "harbour",
    "endeavour",
    "rigour",
    "vigour",
    "odour",
    // -re
    "centre",
    "metre",
    "litre",
    "fibre",
    "calibre",
    "theatre",
    "lustre",
    "manoeuvre",
    // -ce nouns
    "licence",
    "defence",
    "offence",
    "pretence",
    // the doubled -lled
    "modelled",
    "labelled",
    "cancelled",
    "travell",
    "signall",
    "fuelled",
    "totalled",
    "levelled",
    "channelled",
    "dialled",
    // -ogue
    "catalogue",
    "analogue",
    "dialogue",
    // one-offs
    "grey",
    "whilst",
    "amongst",
    "judgement",
    "acknowledgement",
    "learnt",
    "spelt",
    "burnt",
    "dreamt",
    "aluminium",
    "storey",
    "enquire",
    "sceptic",
    "mould",
    "plough",
    "draught",
    "tyre",
    "nought",
    "ageing",
    "cheque",
    "towards",
];

/// Where a British spelling is the correct text: `(path, term)`.
///
/// A quote of another program's output is not this crate's prose. `overdraw_cost.rs` names the
/// message RADV returns when a submission outruns the driver's watchdog -- "the CS has been
/// cancelled because the context is lost" -- and a reader who greps for that string has to find
/// it. Correcting the spelling would make the quote wrong, which is worse than the inconsistency.
///
/// Keyed by term as well as by path, so the exemption is for the word that was argued for rather
/// than for the file: any other British spelling in `overdraw_cost.rs` still fails.
const EXEMPT: &[(&str, &str)] = &[("benches/overdraw_cost.rs", "cancelled")];

/// Directories read, relative to this crate's root.
const ROOTS: &[&str] = &["src", "crates", "tests", "benches", "examples", ".github"];

/// Files at the root that are read. The walk starts below it, so these are named.
const ROOT_FILES: &[&str] = &["README.md", "Cargo.toml", "rust-toolchain.toml"];

/// Directories skipped wherever they appear. Neither is source.
const SKIPPED: &[&str] = &["target", ".git"];

/// Extensions held to the rule: the languages this crate writes prose in.
const PROSE: &[&str] = &["rs", "md", "wgsl", "toml", "yml"];

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if !SKIPPED.contains(&name.as_ref()) {
                walk(&path, out);
            }
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| PROSE.contains(&ext))
        {
            out.push(path);
        }
    }
}

#[test]
fn the_crate_is_written_in_american_english() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<PathBuf> = ROOT_FILES.iter().map(|name| root.join(name)).collect();
    for name in ROOTS {
        walk(&root.join(name), &mut files);
    }
    assert!(
        files.len() > 60,
        "walked {} files, which is too few to be this crate -- the walk is broken, not the \
         spelling",
        files.len()
    );

    // Expanded once rather than per file.
    let mut terms: Vec<String> = BRITISH.iter().map(|word| (*word).to_string()).collect();
    for stem in ISE {
        terms.push(format!("{stem}e"));
        terms.push(format!("{stem}ing"));
        terms.push(format!("{stem}ation"));
    }

    let mut found = Vec::new();
    for path in &files {
        // This file names every word it forbids, so it answers to none of them. A check that has
        // to spell out what it looks for cannot also be looked in.
        if path
            .file_name()
            .is_some_and(|name| name == "american_english.rs")
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let relative = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let lower = text.to_lowercase();
        for term in &terms {
            if lower.contains(term) && !EXEMPT.contains(&(relative.as_str(), term.as_str())) {
                found.push(format!("{relative}: {term}"));
            }
        }
    }
    found.sort();
    found.dedup();

    assert!(
        found.is_empty(),
        "this crate is written in American English, and these are not:\n  {}",
        found.join("\n  ")
    );
}
