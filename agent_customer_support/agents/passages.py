from agent_customer_support import citations as cite


def passages_block(passages: list[str], with_sections: bool = False) -> str:
    """Number the passages for the model, optionally listing each one's headings.

    The heading list turns "name the section you used" from a guess into a choice from a
    closed set. Without it the model reaches for whatever looks most like a title, which
    in this corpus is the summary line prepended to every chunk — not a heading at all,
    so the declaration fails validation and the citation loses its section.

    Reuses `cite.sections`, the same function `cite.select` validates the answer against.
    Sharing that call is the point: the list the model reads and the whitelist it is
    judged by cannot drift apart.

    A passage with no headings gets no annotation rather than an empty one, and the Q&A
    block never asks for annotation — CS records are authored prose and carry no headings.

    Shared by KnowledgeAgent (compose, repair) and IssueVerificationAgent (doc check), so
    both read the guides in the same numbered form.
    """
    out = []
    for i, p in enumerate(passages):
        head = f"[{i}]"
        if with_sections:
            names = cite.sections(p)
            if names:
                head = f"{head} (các mục trong đoạn này: {' | '.join(names)})\n"
        out.append(f"{head} {p}")
    return "\n\n".join(out)
