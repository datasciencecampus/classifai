---
title: Decision Records
date: 2024-07-11
---

This folder contains decision records, which are short markdown documents.
Decision records are for recording the details of any substantial decision that benefits
from group discussion and agreement. This includes architectural decisions, common assumptions,
and decisions about data sources.

An example of an architectural decision might be to use a particular database product.
An assumption might be to hard-code a particular constant into an analytical output.
A data decision might be to use a particular source of data in preference to an alternative.

The purpose of recording decisions is:

* To allow all team members an explicit opportunity to feed in to decision-making and contribute their views
* To maintain a record for quality assurance, so that users and stakeholders can evaluate decisions
* To enable future maintainers and contributors to understand why the codebase is the way that it is.


[The ADR GitHub Organisation](https://adr.github.io/) suggests that a good decision record
should provide the following information:

> In the context of `<use case/user story u>`, facing `<concern c>` we decided for `<option o>` and neglected `<other options>`, to achieve `<system qualities/desired consequences>`, accepting `<downside d/undesired consequences>`, because `<additional rationale>`.

The template provided has these headings.

### Creating a decision record

Create a new decision record at any time, by any member of the team when a decision becomes salient.
Don't wait for the decision to be approved before making a decision record or proferring a preferred decision.
We should have some conventions for allowing discussion and making decisions, but that's
less important than just getting into the habit of writing down the details.

Expect disagreement and open discussion.

Making a decision record doesn't require an issue to be raised, but it can be part of the fulfilment of a 'Research Topic X'-style ticket.

The filename should refer to the problem/context, rather than the preferred solution.
E.g. `choose-embedding-model.md` rather than `use-hugging-face.md`.

### Decision records summary

<!-- TODO: Add an auto-generated table -->
