---
type: project-home
project: llm-council
date: 2026-03-07
cssclasses:
  - project-home
---
# LLM Council
*[[dev-hub|Hub]]*
<span class="hub-status">Functional. Added `council_topfour.py` — reusable 4-model high-effort runner that files synthesis artifacts directly into any project's specs/.</span>

Multi-model LLM deliberation system. Sends queries to Claude, GPT, and Gemini independently, then synthesizes perspectives via a Chairman model.

## Specs

```base
filters:
  and:
    - file.folder.contains("specs/llm-council")
    - type != "spec-prompts"
properties:
  "0":
    name: file.link
    label: Spec
  "1":
    name: type
    label: Type
  "2":
    name: date
    label: Date
  "3":
    name: created_by
    label: Created By
  "4":
    name: file.mtime
    label: Modified
views:
  - type: table
    name: All Specs
    order:
      - type
      - file.name
      - file.mtime
      - file.backlinks
    sort:
      - property: file.mtime
        direction: DESC
      - property: type
        direction: ASC
```

> [!abstract]- Project Plans (`$= dv.pages('"knowledge/plans"').where(p => p.project == "llm-council").length`)
> ```dataview
> TABLE title, default(date, file.ctime) as Date
> FROM "knowledge/plans"
> WHERE project = "llm-council"
> SORT default(date, file.ctime) DESC
> ```

> [!note]- Sessions (`$= dv.pages('"knowledge/sessions/llm-council"').length`)
> ```dataview
> TABLE topic
> FROM "knowledge/sessions/llm-council"
> SORT file.mtime DESC
> LIMIT 5
> ```
>
> > [!note]- All Sessions
> > ```dataview
> > TABLE topic
> > FROM "knowledge/sessions/llm-council"
> > SORT file.mtime DESC
> > ```
