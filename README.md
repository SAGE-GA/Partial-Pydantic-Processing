# Pydantic Partial Library

Utilities for guiding multi-turn AI conversations that gradually fill in [Pydantic](https://docs.pydantic.dev/) models. The helpers wrap a root `UpdatableModel` instance, generate partial schemas that constrain LLM responses to just the fields you want next, and then merge structured output back into the live object without clobbering previously collected data.

## Key Concepts

- **UpdatableModel** – Subclass instead of `BaseModel` when you need to request partial structured completions and merge them into an existing instance.
- **SmartAIInteractionManager** – Builds partial response schemas, predicts where responses belong, and applies them to the backing model while keeping an interaction history.
- **Field specs** – Strings that describe what you want the AI to return next. They can be plain field names (`"title"`), collection names (`"docs"`), dot notation (`"script.video_heard"`), or lists of dot specs (`["script.type", "script.video_heard"]`).

## Quick Start

```python
from typing import List, Optional
from pydantic import BaseModel

from pydantic_partial import UpdatableModel, SmartAIInteractionManager

class ScriptScene(UpdatableModel):
    type: Optional[str] = None
    video_heard: Optional[str] = None
    narration: Optional[str] = None

class CourseDraft(UpdatableModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    script: List[ScriptScene] = []

course = CourseDraft()
manager = SmartAIInteractionManager(course)

# Ask the LLM for the course title only
schema, path_hint = manager.smart_create_ai_schema("title")
ai_response = call_llm(prompt="Give me a title", response_format=schema)  # returns BaseModel instance
manager.smart_apply_ai_response(ai_response.parsed, "title")

# Later, ask for script snippets while preserving existing items
schema, path_hint = manager.smart_create_ai_schema("script.video_heard")
ai_response = call_llm(prompt="Give me an OST description", response_format=schema)
manager.smart_apply_ai_response(ai_response.parsed, "script.video_heard", append_mode=True)

print(course.model_dump())
```

The manager automatically discovers nested models, creates schemas tailored to the requested fields, and merges responses back into `course` without overwriting other values.

## Usage Patterns

### 1. Generate a schema for a single root field

```python
schema, path = manager.create_ai_schema_for_field_name("summary")
ai_response = llm_call(response_format=schema)
manager.apply_ai_response_to_field_name(ai_response.parsed, "summary")
```

- Schema contains only `summary`.
- Updates `course.summary` while leaving other fields intact.

### 2. Address nested fields by path

```python
schema = manager.create_ai_schema_for_field_path("script.0.narration")
ai_response = llm_call(response_format=schema)
manager.apply_ai_response_to_field_path(ai_response.parsed, "script.0.narration")
```

- Works with dot notation `collection.index.field`.
- Automatically creates missing list items before merging.

### 3. Append complete objects to a collection

```python
schema = manager.create_ai_schema_for_collection_item("script")
ai_response = llm_call(response_format=schema)
manager.append_ai_response_to_collection(ai_response.parsed, "script")
```

- Use when the LLM should return fully populated objects for a list.

### 4. Request partial objects for arrays

```python
schema, _ = manager.smart_create_ai_schema("script.video_heard")
ai_response = llm_call(response_format=schema)
manager.apply_array_partial_response(ai_response.parsed, "script", "video_heard", append_mode=False)
```

- Default overwrites the collection while preserving other fields on existing items.
- Set `append_mode=True` to extend instead of replacing.

### 5. Multi-field partial updates

```python
schema, _ = manager.smart_create_ai_schema(["script.type", "script.video_heard"])
ai_response = llm_call(response_format=schema)
manager.smart_apply_ai_response(ai_response.parsed, ["script.type", "script.video_heard"], append_mode=False)
```

- Schema asks for both `type` and `video_heard` in each array item.
- Overwrite vs append behavior controlled by `append_mode`.

### 6. Multiple scalar fields at once

```python
schema, _ = manager.smart_create_ai_schema(["title", "summary"])
ai_response = llm_call(response_format=schema)
manager.smart_apply_ai_response(ai_response.parsed, ["title", "summary"])
```

- Produces a partial schema covering both fields and merges in one call.

## API Reference

### `class UpdatableModel(BaseModel)`

| Method | Description | Parameters |
| --- | --- | --- |
| `create_partial_schema(fields_to_fill: List[str]) -> Type[BaseModel]` | Dynamically build a Pydantic model containing only the requested fields. | `fields_to_fill` – list of field names defined on the model. Unknown fields are ignored. |
| `merge_partial_update(partial_instance: BaseModel, fields_updated: List[str]) -> UpdatableModel` | Returns a copy with the provided fields merged from a partial model (ignores `None`). | `partial_instance` – parsed LLM response; `fields_updated` – fields to copy from the partial payload. |

### `class SmartAIInteractionManager`

Construct with `SmartAIInteractionManager(main_model: UpdatableModel)` where `main_model` is the instance you are progressively filling. The manager records every application in `interaction_history` (list of dicts with `field_path`, `ai_response`, `timestamp`).

#### Schema generation helpers

| Method | Purpose | Key Parameters and Flags |
| --- | --- | --- |
| `create_ai_schema_for_collection_item(collection_name)` | Return the full item schema for a list field so the AI can generate complete objects. | `collection_name` – list field defined on the root model. Raises `ValueError` if unknown. |
| `create_ai_schema_for_field_path(field_path)` | Build a partial schema from a dot path like `"script.0.narration"`. | `field_path` – either a root field (`"title"`) or `collection.index.field`. Currently supports depth of 0 or 1 level into list items. |
| `create_ai_schema_for_field_name(field_name)` | Locate a field anywhere in the root or nested list models and return its partial schema. | `field_name` – name without path. Auto-detects append position and returns `(schema, resolved_path)`. |
| `smart_create_ai_schema(field_specs)` | High-level dispatcher that understands scalar fields, whole collections, dot notation, and multi-field specs. | `field_specs` – string or list. Returns `(schema, predicted_path)`; for arrays uses `[].field` syntax in the hint. |
| `create_array_partial_schema(collection_name, field_name)` | Schema shaped like `{collection_name: [{field_name: ...}]}` for single-field array partials. | `field_name` – field on the collection item model. |
| `create_array_partial_schema_multi_field(collection_name, field_names)` | Same as above but multiple fields per item. | `field_names` – list of fields in the collection's item model. |

#### Applying AI responses

| Method | Purpose | Parameters & Flags |
| --- | --- | --- |
| `apply_ai_response_to_field_path(ai_response, field_path)` | Merge a parsed response into a specific field path. Creates missing list items automatically. | `ai_response` – result parsed by the schema produced earlier. `field_path` – dot notation as above. |
| `apply_ai_response_to_field_name(ai_response, field_name, append_mode=None)` | Auto-resolve where a field name belongs and merge it. | `append_mode` – `True` to append to lists, `False` to overwrite index `0`, `None` to auto-detect (append when collection already has data). |
| `append_ai_response_to_collection(ai_response, collection_name)` | Append a full item to a list field. Converts dictionaries into the correct model automatically. | `ai_response` – either already of the correct Pydantic type or a structurally matching model; `collection_name` – target list. |
| `apply_array_partial_response(ai_response, collection_name, field_name, append_mode=False)` | Apply partial objects for a single field in a list. Preserves untouched fields on existing items. | `append_mode` – `True` to extend the list, `False` to overwrite from index 0. |
| `apply_array_partial_response_multi_field(ai_response, collection_name, field_names, append_mode=False)` | Same as above but updates multiple fields per item. | `field_names` – list of field names to copy from each item. |
| `smart_apply_ai_response(ai_response, field_specs, append_mode=False)` | High-level counterpart to `smart_create_ai_schema`. Accepts the same field spec formats and routes to the proper merge helper. | `field_specs` – string or list; `append_mode` – only used for dot notation specs to toggle append vs overwrite semantics. |

#### Utility methods

| Method | Notes |
| --- | --- |
| `_ensure_collection_item_exists(collection_name, index)` | Internal helper that extends list fields with empty model instances when you target a specific index. |
| `_get_all_nested_model_info()` & `_get_model_class_for_collection(collection_name)` | Discover nested `UpdatableModel` classes automatically. Works for list and `Optional[List[...]]` annotations. |
| `_log_interaction(field_path, ai_response)` | Adds an entry to `interaction_history` so you can audit what was applied when. |

## Working With an LLM

1. **Create a schema** using `smart_create_ai_schema` or one of the lower-level helpers.
2. **Send the schema** to your LLM inference endpoint as a structured response format.
3. **Parse the response** into the returned schema class (most SDKs do this automatically when you pass `response_format`).
4. **Apply the response** using the matching `smart_apply_ai_response` or specific helper.

Repeat the cycle to progressively collect every field of your Pydantic model across conversational turns without manually tracking state.

## Tips

- Always subclass `UpdatableModel` for both root and nested models so merge helpers are available.
- When you want the LLM to *append* items (e.g., new script beats), pass `append_mode=True` when applying array partials.
- The schema helpers ignore unknown fields gracefully, so you can reuse them even when your models evolve.
- Inspect `manager.interaction_history` for debugging or analytics.

