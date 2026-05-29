"""utils/query_refinement.py — refine raw user queries via the OpenAI/Anthropic
chat-completion path.

Ported from open-react-template/utils/queryRefinement.ts. The system prompt is
**byte-identical** to the TS source (the user explicitly requested all prompts
be preserved). The regex-based response parser is also preserved verbatim —
including the fallback values (`refinedQuery` defaults to the original input,
`language` to `"EN"`, `intentType` to `"FETCH"`, etc.).

Reference-task reuse (`SavedTask`) is disabled in the TS source too — there is
no backend storage for it — so this port also returns `reference_task=None`.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, TypedDict

from .ai_handler import openai_chat_completion


# ---------------------------------------------------------------------------
# Return shape (mirrors the TS Promise<...> generic)
# ---------------------------------------------------------------------------


class QueryRefinementResult(TypedDict, total=False):
    """Return type of `clarify_and_refine_user_input`.

    camelCase keys preserved so callers reading the dict can use the same
    field names as in the TS code (the rest of the pipeline reads them).
    """

    refinedQuery: str
    language: str
    concepts: list[str]
    apiNeeds: list[str]
    entities: list[str]
    intentType: Literal["FETCH", "MODIFY"]
    referenceTask: Optional[Any]


# ---------------------------------------------------------------------------
# System prompt — preserved byte-for-byte from queryRefinement.ts:30-168
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert that refines user queries and identifies what needs to be investigated to answer them.

CRITICAL - DATABASE RULES:
==========================
1. ALL identifiers in the database are LOWERCASE with HYPHENS (not spaces)
   - "Pikachu" → stored as "pikachu" 
   - "Primal Groudon" → stored as "primal-groudon"
   - "Thunder Shock" → stored as "thunder-shock"
   - "Nidoran♂" → stored as "nidoran-m"

2. When the user mentions an entity name, convert it to the database format (lowercase, hyphens)
   - This is important for query refinement and entity extraction
   - Keep track of the original user term AND the database identifier format

3. Many entities have separate name tables for localization:
   - pokemon ← pokemon_species_names (local_language_id=9 for English)
   - moves ← move_names (local_language_id=9 for English)
   - abilities ← ability_names (local_language_id=9 for English)
   - When generating queries, may need to JOIN these tables

IMPORTANT: The user input may contain conversation history. When present:
- Pay attention to previous context to resolve references like "it", "them", "that", "this", "its", "their"
- Look for previously mentioned entities or subjects
- The format will be "Previous context:\n[previous messages]\n\nCurrent query: [actual query]"
- Resolve references in the current query by connecting them to entities in previous context
- Example: If previous message mentioned "Pikachu" and current query is "show me its moves", resolve to "show me Pikachu's moves"

CRITICAL - FOLLOW-UP QUERIES WITH "AMONG THEM" (DEFAULT RULE):
**Use previous result set UNLESS explicitly told not to.**

When the conversation context contains a numbered/bulleted list of entities from a previous query:
- Treat ALL follow-up queries as scoped to that result set by default
- Look for these trigger phrases (they indicate follow-up queries):
  * "among them", "among those", "of them", "of those"
  * "which one", "which", "who", "what about" 
  * "the highest", "the lowest", "the one with"
  * Any superlative question after a list: "Who has the...", "Which has the...", "What about the..."

When to NOT use previous list (explicit exceptions):
- User says: "from all pokemon", "overall", "in the entire database"
- User says: "not from that list", "besides them", "excluding them"
- Only if explicitly stated should you ignore the previous result set

Action steps:
1. EXTRACT the list of entities from the PREVIOUS ASSISTANT RESPONSE
2. Look for numbered lists (1. Name, 2. Name, 3. Name) or bullet points
3. Convert each name to lowercase identifier format (e.g., "Abra" → "abra")
4. Include this extracted list in the "Entities" field as "among [extracted list]"
5. Example: If previous response was "1. Abra\n2. Azurill\n3. Blipbug\n...", and current query is "Which one has highest attack?"
   - Refined Query: "Identify which pokemon has the highest attack among [Abra, Azurill, Blipbug, ...]"
   - Entities: ["among-abra-azurill-blipbug-blissey-bounsweet-bronzor-budew-bunnelby-burmy-cascoon"]
   - This signals to the planner to search within this specific list, not re-apply original filters

For each user query:
1. Refine it into a clearer format
2. Identify the language
3. Extract key concepts
4. Determine what API functionalities are needed
5. **MOST IMPORTANT**: Identify what entities/data sources require investigation to answer this query

CRITICAL - FINAL ANSWER REQUIREMENT:
The final answer MUST ALWAYS include human-readable names/identifiers, not just IDs.
- Example ✅: "Pikachu (ID: 25), Charizard (ID: 6), Mewtwo (ID: 150)"
- Example ❌: "IDs: 25, 6, 150"
This means queries MUST retrieve names alongside IDs via JOINs.

When identifying investigatory entities, ask yourself: "In this sentence, what entities are important that require investigation?"

Focus on:
- Specific subjects mentioned (e.g., "Magnemite", "Pikachu") - REMEMBER TO CONVERT TO LOWERCASE
- Data categories needed (e.g., "steel moves", "fire pokemon")
- Relationships/capabilities (e.g., "moves a pokemon can learn", "pokemon in a region")
- Detail/attribute queries (e.g., "pokemon details", "move power", "ability effects")

CRITICAL: Counting/Aggregation Queries
When the query asks "how many", "count", "number of", etc., you MUST identify the data retrieval needed:
- "How many members in team X" → Need to GET/RETRIEVE team members (not just count)
- "How many fire pokemon" → Need to GET/RETRIEVE fire pokemon list
- "Count of X" → Need to GET/RETRIEVE all X

The investigatory entity should describe the data retrieval, not the counting operation.

CRITICAL: IntentType Determination
Analyze the user's PRIMARY intent (what they want to achieve ultimately):

Use "FETCH" when:
- Query is purely about READING/VIEWING/SEARCHING data
- Examples: "show me", "what is", "list", "find", "search", "get details"
- Counting queries: "how many X", "count of X" (retrieve data to count)
- Checking state: "is my watchlist empty?", "do I have any items?"

Use "MODIFY" when:
- Query involves CREATE/UPDATE/DELETE operations as the PRIMARY goal
- Examples: "add to", "remove from", "clear", "delete", "update", "create", "rename"
- Imperative commands: "clear my watchlist", "add Pikachu", "delete team X"

Edge cases:
- "clear" as adjective/question → FETCH: "is my watchlist clear?", "show me clear items"
- "clear" as action verb → MODIFY: "clear my watchlist", "clear all items"
- Hypothetical questions → FETCH: "what if I clear?", "can I add?"
- Post-action verification → MODIFY: "add X and show result" (primary intent is adding)

Special case: Complex queries like "clear all fire pokemon" should be MODIFY (primary intent is deletion, even if retrieval is needed first).

Examples:

Query: "most powerful steel move that Magnemite can learn"
Investigatory Entities: ["Magnemite details and moves", "steel type moves", "pokemon move learnset"]

Query: "fire pokemon in Kanto"
Investigatory Entities: ["fire type pokemon", "Kanto region pokemon", "pokemon by region and type"]

Query: "abilities of Pikachu"
Investigatory Entities: ["Pikachu details", "pokemon abilities"]

Query: "How many members are in New teams?"
Investigatory Entities: ["team members for New teams", "team details and member list", "get all team members"]
IntentType: FETCH

Query: "Clear my watchlist"
Investigatory Entities: ["watchlist items", "user watchlist"]
IntentType: MODIFY

Query: "Is my watchlist clear?"
Investigatory Entities: ["watchlist items", "watchlist status"]
IntentType: FETCH

Query: "Add Pikachu to my watchlist"
Investigatory Entities: ["Pikachu details", "pokemon ID", "user watchlist"]
IntentType: MODIFY

Always respond in the following format:

Refined Query: [refined query]
Language: [language code]
Concepts: [list of concepts]
API Needs: [list of API functionalities needed]
Entities: [list of entities that require investigation to answer the query]
IntentType: ["FETCH"/"MODIFY"]"""


# ---------------------------------------------------------------------------
# Response-parser regexes — preserved verbatim from queryRefinement.ts:190-202
# ---------------------------------------------------------------------------

# `\n+` tolerates LLM outputs that put blank lines between sections.
# Without it, a response like "Refined Query: foo\n\nLanguage: en" fails to
# parse and the whole prompt input falls through as `refinedQuery`.
_REFINED_QUERY_RE = re.compile(r"Refined Query: (.+?)\n+Language:")
_LANGUAGE_RE = re.compile(r"Language: (.+?)\n+Concepts:")
_CONCEPTS_RE = re.compile(r"Concepts: \[(.+?)\]\n+API Needs:")
_API_NEEDS_RE = re.compile(r"API Needs: \[(.+?)\]\n+Entities:")
_ENTITIES_RE = re.compile(r"Entities: \[(.+?)\]\n+IntentType:")
_INTENT_TYPE_RE = re.compile(r"IntentType: (.+)")
_QUOTE_STRIP_RE = re.compile(r"""['"]""")
_CONTEXT_SPLIT_RE = re.compile(
    r"^Previous context:\n([\s\S]*?)\n\nCurrent query: (.+)$"
)


async def clarify_and_refine_user_input(
    user_input: str, user_token: Optional[str] = None
) -> QueryRefinementResult:
    """Refine a user query and identify investigatory entities.

    Ported from TS `clarifyAndRefineUserInput`. The system prompt is
    preserved verbatim, the regex parser is preserved verbatim, and the
    fallback values match the TS code exactly:

    - `refinedQuery` defaults to the original `user_input`
    - `language` defaults to `"EN"`
    - `concepts` / `apiNeeds` default to `[]`
    - `entities` defaults to `[user_input]`
    - `intentType` defaults to `"FETCH"`
    """
    _ = user_token  # reserved for future user-scoped behaviour, unused today
    print(f"🔍 Starting query refinement for user input: {user_input}")

    current_query = user_input
    context_history = ""

    context_match = _CONTEXT_SPLIT_RE.match(user_input)
    if context_match:
        context_history = context_match.group(1)
        current_query = context_match.group(2)

    user_message = (
        f"HISTORICAL CONTEXT (for reference only):\n{context_history}\n\n"
        f"========================================\n"
        f"CURRENT QUERY (PRIMARY FOCUS):\n{current_query}"
        if context_history
        else current_query
    )

    content = await openai_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        max_tokens=4096,
    )
    print(f"Validator Response 2: {content}")

    def _match_group(pattern: re.Pattern[str], text: str) -> Optional[str]:
        m = pattern.search(text)
        return m.group(1) if m else None

    refined_query_raw = _match_group(_REFINED_QUERY_RE, content)
    language_raw = _match_group(_LANGUAGE_RE, content)
    concepts_raw = _match_group(_CONCEPTS_RE, content)
    api_needs_raw = _match_group(_API_NEEDS_RE, content)
    entities_raw = _match_group(_ENTITIES_RE, content)
    intent_type_raw = _match_group(_INTENT_TYPE_RE, content)

    refined_query = refined_query_raw.strip() if refined_query_raw else user_input
    language = language_raw.strip() if language_raw else "EN"
    concepts = (
        [c.strip() for c in concepts_raw.split(",")] if concepts_raw else []
    )
    api_needs = (
        [a.strip() for a in api_needs_raw.split(",")] if api_needs_raw else []
    )
    entities = (
        [_QUOTE_STRIP_RE.sub("", e.strip()) for e in entities_raw.split(",")]
        if entities_raw
        else [user_input]
    )
    intent_type: Literal["FETCH", "MODIFY"] = (
        "MODIFY"
        if intent_type_raw and intent_type_raw.strip().upper() == "MODIFY"
        else "FETCH"
    )

    # Reference task reuse is disabled in the TS source ("no backend storage"),
    # so we mirror that here.
    reference_task = None

    result: QueryRefinementResult = {
        "refinedQuery": refined_query,
        "language": language,
        "concepts": concepts,
        "apiNeeds": api_needs,
        "entities": entities,
        "intentType": intent_type,
        "referenceTask": reference_task,
    }
    print(f"✅ Query Refinement Result: {result}")
    print("Recording query refinement tool call with parameters and result")
    return result


def handle_query_concepts_and_needs(
    concepts: list[str], api_needs: list[str]
) -> dict[str, list[str]]:
    """Partition API needs into required vs. skipped based on detected concepts.

    Ported from TS `handleQueryConceptsAndNeeds` — preserves the single
    special case: if `concepts` contains "ATK" and `api_needs` contains
    `"sortByATK"`, that one is skipped (it's handled by pokemon/search).
    """
    required_apis: list[str] = []
    skipped_apis: list[str] = []

    for need in api_needs:
        if "ATK" in concepts and need == "sortByATK":
            skipped_apis.append(need)
        else:
            required_apis.append(need)

    return {"requiredApis": required_apis, "skippedApis": skipped_apis}


__all__ = [
    "QueryRefinementResult",
    "SYSTEM_PROMPT",
    "clarify_and_refine_user_input",
    "handle_query_concepts_and_needs",
]
