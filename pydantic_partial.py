"""
Pydantic Partial Library - Multi-turn AI interaction support for Pydantic models.

This library enables you to fill Pydantic models across multiple AI turns without overwrites
by creating partial schemas that constrain AI responses to specific fields.

Usage:
    from pydantic_partial import UpdatableModel, SmartAIInteractionManager

    # Your models inherit from UpdatableModel instead of BaseModel
    class MyModel(UpdatableModel):
        name: str = None
        items: List[Item] = []

    # Create manager and use field names or paths
    model = MyModel()
    manager = SmartAIInteractionManager(model)

    # Method 1: Just field names (auto-discovery)
    schema, path = manager.create_ai_schema_for_field_name("name")
    ai_response = your_ai_call(response_format=schema)
    manager.apply_ai_response_to_field_name(ai_response.parsed, "name")

    # Method 2: Explicit field paths
    schema = manager.create_ai_schema_for_field_path("items.0.description")
    ai_response = your_ai_call(response_format=schema)
    manager.apply_ai_response_to_field_path(ai_response.parsed, "items.0.description")
"""

from typing import Any, Dict, List, Type, TypeVar, Union

from pydantic import BaseModel, create_model

# Generic type for UpdatableModel
T = TypeVar("T", bound="UpdatableModel")


class UpdatableModel(BaseModel):
    """
    Base class that provides dynamic schema generation for partial updates.
    Use this instead of BaseModel for models that need multi-turn AI support.
    """

    @classmethod
    def create_partial_schema(cls, fields_to_fill: List[str]) -> Type[BaseModel]:
        """Create a new Pydantic model with only the specified fields."""
        field_definitions = {}

        for field_name in fields_to_fill:
            if field_name in cls.model_fields:
                original_field = cls.model_fields[field_name]
                field_definitions[field_name] = (
                    original_field.annotation,
                    original_field,
                )

        model_name = f"{cls.__name__}Partial_{('_'.join(fields_to_fill))}"
        return create_model(model_name, **field_definitions)

    def merge_partial_update(
        self: T, partial_instance: BaseModel, fields_updated: List[str]
    ) -> T:
        """Merge a partial update back into this model"""
        updates = {}
        partial_data = partial_instance.model_dump()

        for field_name in fields_updated:
            if field_name in partial_data and partial_data[field_name] is not None:
                updates[field_name] = partial_data[field_name]

        return self.model_copy(update=updates)


class SmartAIInteractionManager:
    """
    Manages AI interactions with Pydantic models, providing automatic field discovery,
    partial schema generation, and data preservation across multiple turns.
    """

    def __init__(self, main_model: UpdatableModel):
        self.main_model = main_model
        self.interaction_history: List[Dict[str, Any]] = []

    def create_ai_schema_for_collection_item(
        self, collection_name: str
    ) -> Type[BaseModel]:
        """
        Create a schema for AI structured output for a complete collection item.
        Use this when you want the AI to return a complete object to append to an array.

        Args:
            collection_name: Name of the collection field (e.g., "docs", "items")

        Returns:
            The full schema for the collection's item type
        """
        if collection_name not in self.main_model.__class__.model_fields:
            raise ValueError(f"Collection '{collection_name}' not found in model")

        model_class = self._get_model_class_for_collection(collection_name)
        return model_class

    def create_ai_schema_for_field_path(self, field_path: str) -> Type[BaseModel]:
        """
        Create a schema for AI structured output from a field path.

        Args:
            field_path: Dot notation path like "script.0.video_heard" or "video_heard"

        Returns:
            A partial schema containing only the target field
        """
        path_parts = field_path.split(".")

        if len(path_parts) == 1:
            # Root level field like "code"
            field_name = path_parts[0]
            return self.main_model.__class__.create_partial_schema([field_name])

        elif len(path_parts) == 3:
            # Nested field like "script.0.video_heard"
            collection_name, index_str, field_name = path_parts
            target_model_class = self._get_model_class_for_collection(collection_name)
            return target_model_class.create_partial_schema([field_name])

        else:
            raise ValueError(f"Unsupported field path format: {field_path}")

    def create_ai_schema_for_field_name(
        self, field_name: str
    ) -> tuple[Type[BaseModel], str]:
        """
        Create a schema for AI structured output from just a field name.
        Automatically finds which model contains this field.

        Args:
            field_name: Just the field name like "video_heard"

        Returns:
            Tuple of (partial_schema, resolved_field_path)
        """
        # Check if it's in the root model
        if field_name in self.main_model.__class__.model_fields:
            schema = self.main_model.__class__.create_partial_schema([field_name])
            return schema, field_name

        # Check nested models
        nested_models = self._get_all_nested_model_info()

        for collection_name, model_class in nested_models.items():
            if field_name in model_class.model_fields:
                schema = model_class.create_partial_schema([field_name])

                # Use same auto-detection logic as apply_ai_response_to_field_name
                collection = getattr(self.main_model, collection_name)
                has_existing_data = False

                for item in collection:
                    # Check if any field in the item has non-default data
                    item_data = item.model_dump()
                    if any(value not in [None, "", []] for value in item_data.values()):
                        has_existing_data = True
                        break

                # Predict the correct path based on auto-detection
                if has_existing_data:
                    next_index = len(collection)
                    resolved_path = f"{collection_name}.{next_index}.{field_name}"
                else:
                    resolved_path = f"{collection_name}.0.{field_name}"

                return schema, resolved_path

        raise ValueError(f"Field '{field_name}' not found in any model")

    def apply_ai_response_to_field_name(
        self, ai_response: BaseModel, field_name: str, append_mode: bool = None
    ) -> str:
        """
        Apply AI response using just a field name.
        Automatically determines where to put it.

        Args:
            ai_response: The AI response
            field_name: Just the field name like "video_heard"
            append_mode: If True, append to arrays. If False, overwrite index 0.
                        If None (default), auto-detect: append for array fields, overwrite for scalar fields.

        Returns:
            The resolved field path where the data was applied
        """

        # Check if it's in the root model
        if field_name in self.main_model.__class__.model_fields:
            self.main_model = self.main_model.merge_partial_update(
                ai_response, [field_name]
            )
            self._log_interaction(field_name, ai_response)
            return field_name

        # Check nested models
        nested_models = self._get_all_nested_model_info()

        for collection_name, model_class in nested_models.items():
            if field_name in model_class.model_fields:
                # Auto-detect append mode if not specified
                if append_mode is None:
                    # Check if this collection already has items with data
                    collection = getattr(self.main_model, collection_name)
                    has_existing_data = False

                    for item in collection:
                        # Check if any field in the item has non-default data
                        item_data = item.model_dump()
                        if any(
                            value not in [None, "", []] for value in item_data.values()
                        ):
                            has_existing_data = True
                            break

                    # Default: append if there's existing data, otherwise use index 0
                    append_mode = has_existing_data

                if append_mode:
                    # Find the next available index
                    collection = getattr(self.main_model, collection_name)
                    next_index = len(collection)
                    self._ensure_collection_item_exists(collection_name, next_index)
                    resolved_path = f"{collection_name}.{next_index}.{field_name}"
                else:
                    # Use index 0 by default, create if needed
                    self._ensure_collection_item_exists(collection_name, 0)
                    resolved_path = f"{collection_name}.0.{field_name}"

                # Apply the update
                self.apply_ai_response_to_field_path(ai_response, resolved_path)
                return resolved_path

        raise ValueError(f"Field '{field_name}' not found in any model")

    def smart_apply_ai_response(
        self, ai_response: BaseModel, field_specs, append_mode: bool = False
    ) -> str:
        """
        Single smart function that applies AI responses based on field specification format.

        Patterns:
        - "docs" → Append complete Doc object to docs array
        - "code" → Update single scalar field
        - "script.video_heard" → OVERWRITE array with partial ScriptScene objects (default)
        - ["script.type", "script.video_heard"] → OVERWRITE array with multi-field partial objects

        Args:
            ai_response: The AI response
            field_specs: Field specification(s) - can be string or list of strings
            append_mode: For dot notation, True=append to array, False=overwrite array (default)

        Returns:
            The resolved field path where the data was applied
        """
        # Handle both single string and list inputs
        if isinstance(field_specs, str):
            field_specs_list = [field_specs]
            single_field = True
        else:
            field_specs_list = field_specs
            single_field = False

        # If multiple field specs, handle multi-field application
        if not single_field:
            return self._apply_multi_field_response(
                field_specs_list, ai_response, append_mode
            )

        # Single field spec - use existing logic
        field_spec = field_specs_list[0]

        # Pattern 3: collection.field format (e.g., "script.video_heard")
        # → Apply array of partial objects (overwrite by default, append if requested)
        if "." in field_spec and not field_spec.count(".") > 1:
            collection_name, field_name = field_spec.split(".")
            return self.apply_array_partial_response(
                ai_response, collection_name, field_name, append_mode
            )

        # Pattern 1: Array field (e.g., "docs")
        # → Append complete object
        if field_spec in self.main_model.__class__.model_fields:
            field_info = self.main_model.__class__.model_fields[field_spec]

            if (
                hasattr(field_info.annotation, "__origin__")
                and field_info.annotation.__origin__ is list
            ):
                # Handle wrapped AI responses (like {"docs": [{"type": "...", "content": "..."}]})
                ai_data = ai_response.model_dump()
                if (
                    field_spec in ai_data
                    and isinstance(ai_data[field_spec], list)
                    and len(ai_data[field_spec]) > 0
                ):
                    # Extract the actual object from the wrapper
                    actual_item_data = ai_data[field_spec][0]
                    model_class = self._get_model_class_for_collection(field_spec)
                    actual_item = model_class(**actual_item_data)
                    return self.append_ai_response_to_collection(
                        actual_item, field_spec
                    )
                else:
                    # Direct object structure
                    return self.append_ai_response_to_collection(
                        ai_response, field_spec
                    )
            else:
                # Pattern 2: Scalar field (e.g., "code")
                # → Update single field
                self.main_model = self.main_model.merge_partial_update(
                    ai_response, [field_spec]
                )
                self._log_interaction(field_spec, ai_response)
                return field_spec

        # Fallback: try nested field logic for single field updates
        return self.apply_ai_response_to_field_name(ai_response, field_spec)

    def _apply_multi_field_response(
        self,
        field_specs_list: List[str],
        ai_response: BaseModel,
        append_mode: bool = False,
    ) -> str:
        """Apply AI response for multiple field specifications."""

        # Analyze field patterns
        dot_notation_specs = []
        scalar_specs = []

        for spec in field_specs_list:
            if "." in spec and not spec.count(".") > 1:
                dot_notation_specs.append(spec)
            else:
                scalar_specs.append(spec)

        # Handle different combinations
        if dot_notation_specs and not scalar_specs:
            # All dot notation - apply multi-field array partial
            return self._apply_multi_field_array_partial(
                dot_notation_specs, ai_response, append_mode
            )
        elif scalar_specs and not dot_notation_specs:
            # All scalar fields
            return self._apply_multi_scalar_fields(scalar_specs, ai_response)
        else:
            raise ValueError(f"Mixed field patterns not supported: {field_specs_list}")

    def _apply_multi_field_array_partial(
        self,
        dot_notation_specs: List[str],
        ai_response: BaseModel,
        append_mode: bool = False,
    ) -> str:
        """Apply multi-field array partial like ['script.type', 'script.video_heard']"""
        # Extract collection names and field names
        collection_fields = {}
        for spec in dot_notation_specs:
            collection_name, field_name = spec.split(".")
            if collection_name not in collection_fields:
                collection_fields[collection_name] = []
            collection_fields[collection_name].append(field_name)

        # Must all be from same collection
        if len(collection_fields) != 1:
            raise ValueError(
                f"Multi-field array partials must be from same collection: {dot_notation_specs}"
            )

        collection_name = list(collection_fields.keys())[0]
        field_names = collection_fields[collection_name]

        return self.apply_array_partial_response_multi_field(
            ai_response, collection_name, field_names, append_mode
        )

    def _apply_multi_scalar_fields(
        self, scalar_specs: List[str], ai_response: BaseModel
    ) -> str:
        """Apply multiple scalar fields like ['code', 'title']"""
        self.main_model = self.main_model.merge_partial_update(
            ai_response, scalar_specs
        )
        self._log_interaction("+".join(scalar_specs), ai_response)
        return "+".join(scalar_specs)

    def apply_array_partial_response_multi_field(
        self,
        ai_response: BaseModel,
        collection_name: str,
        field_names: List[str],
        append_mode: bool = False,
    ) -> str:
        """
        Apply AI response containing array of partial objects with multiple fields.

        Args:
            ai_response: AI response like {"script": [{"type": "...", "video_heard": "..."}, ...]}
            collection_name: e.g., "script"
            field_names: e.g., ["type", "video_heard"]
            append_mode: If True, append new items. If False (default), overwrite array

        Returns:
            Path where data was applied
        """
        ai_data = ai_response.model_dump()

        if collection_name not in ai_data:
            raise ValueError(f"Expected '{collection_name}' in AI response")

        items_data = ai_data[collection_name]
        if not isinstance(items_data, list):
            raise ValueError(f"Expected array for '{collection_name}' in AI response")

        # Get current collection
        collection = getattr(self.main_model, collection_name)
        model_class = self._get_model_class_for_collection(collection_name)

        if append_mode:
            # APPEND MODE: Add new items to the end
            starting_index = len(collection)

            for item_data in items_data:
                # Create partial item with multiple fields
                partial_data = {
                    field_name: item_data.get(field_name, "")
                    for field_name in field_names
                }
                partial_item = model_class(**partial_data)
                collection.append(partial_item)

            path_range = f"{collection_name}[{starting_index}:{starting_index + len(items_data)}].{'+'.join(field_names)}"
        else:
            # OVERWRITE MODE (default): Replace array contents while preserving other fields

            # Preserve existing items' other fields while updating target fields
            preserved_items = []
            for i in range(min(len(collection), len(items_data))):
                current_item = collection[i]
                preserved_data = current_item.model_dump()
                # Update the target fields with new values
                for field_name in field_names:
                    preserved_data[field_name] = items_data[i].get(field_name, "")
                preserved_items.append(model_class(**preserved_data))

            # Add any additional new items beyond existing array length
            for i in range(len(collection), len(items_data)):
                partial_data = {
                    field_name: items_data[i].get(field_name, "")
                    for field_name in field_names
                }
                partial_item = model_class(**partial_data)
                preserved_items.append(partial_item)

            # Replace the entire collection
            collection.clear()
            collection.extend(preserved_items)

            path_range = (
                f"{collection_name}[0:{len(items_data)}].{'+'.join(field_names)}"
            )

        # Log the interaction
        self._log_interaction(path_range, ai_response)
        return path_range

    def smart_create_ai_schema(self, field_specs) -> tuple[Type[BaseModel], str]:
        """
        Single smart function that creates AI schemas based on field specification format.

        Patterns:
        - "docs" → Complete Doc objects for appending to docs array
        - "code" → Single scalar field update
        - "script.video_heard" → Array of partial ScriptScene objects with just video_heard
        - ["script.type", "script.video_heard"] → Array of partial ScriptScene objects with both fields

        Args:
            field_specs: Field specification(s) - can be string or list of strings

        Returns:
            Tuple of (schema, predicted_path)
        """
        # Handle both single string and list inputs
        if isinstance(field_specs, str):
            field_specs_list = [field_specs]
            single_field = True
        else:
            field_specs_list = field_specs
            single_field = False

        # If multiple field specs, handle multi-field creation
        if not single_field:
            return self._create_multi_field_schema(field_specs_list)

        # Single field spec - use existing logic
        field_spec = field_specs_list[0]

        # Pattern 3: collection.field format (e.g., "script.video_heard")
        # → Array of partial objects with single field
        if "." in field_spec and not field_spec.count(".") > 1:
            collection_name, field_name = field_spec.split(".")
            return self.create_array_partial_schema(
                collection_name, field_name
            ), f"{collection_name}[].{field_name}"

        # Pattern 1: Array field (e.g., "docs")
        # → Complete objects for appending
        if field_spec in self.main_model.__class__.model_fields:
            field_info = self.main_model.__class__.model_fields[field_spec]
            annotation = field_info.annotation

            # Handle Optional[List[...]] by extracting the inner type
            if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                # Check if this is Optional[List[...]] (Union[List[...], None])
                non_none_args = [
                    arg for arg in annotation.__args__ if arg is not type(None)
                ]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                schema = self.create_ai_schema_for_collection_item(field_spec)
                collection = getattr(self.main_model, field_spec)
                predicted_path = f"{field_spec}.{len(collection)}"
                return schema, predicted_path
            else:
                # Pattern 2: Scalar field (e.g., "code")
                # → Single field update
                schema = self.main_model.__class__.create_partial_schema([field_spec])
                return schema, field_spec

        # Fallback: try nested field logic for single field updates
        return self.create_ai_schema_for_field_name(field_spec)

    def _create_multi_field_schema(
        self, field_specs_list: List[str]
    ) -> tuple[Type[BaseModel], str]:
        """
        Create schema for multiple field specifications.

        Examples:
        - ["script.type", "script.video_heard"] → Array of ScriptScene with both fields
        - ["code", "docs"] → Error - mixed patterns not supported
        - ["video_heard", "video_seen"] → Partial schema with both fields (if in same model)
        """

        # Analyze field patterns
        dot_notation_specs = []
        scalar_specs = []
        array_specs = []

        for spec in field_specs_list:
            if "." in spec and not spec.count(".") > 1:
                dot_notation_specs.append(spec)
            elif spec in self.main_model.__class__.model_fields:
                field_info = self.main_model.__class__.model_fields[spec]
                annotation = field_info.annotation

                # Handle Optional[List[...]]
                if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                    non_none_args = [
                        arg for arg in annotation.__args__ if arg is not type(None)
                    ]
                    if len(non_none_args) == 1:
                        annotation = non_none_args[0]

                if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                    array_specs.append(spec)
                else:
                    scalar_specs.append(spec)
            else:
                scalar_specs.append(spec)  # Assume scalar for fallback

        # Handle different combinations
        if dot_notation_specs and not scalar_specs and not array_specs:
            # All dot notation - check if same collection
            return self._create_multi_field_array_partial_schema(dot_notation_specs)
        elif scalar_specs and not dot_notation_specs and not array_specs:
            # All scalar fields - create partial schema
            return self._create_multi_scalar_schema(scalar_specs)
        else:
            raise ValueError(f"Mixed field patterns not supported: {field_specs_list}")

    def _create_multi_field_array_partial_schema(
        self, dot_notation_specs: List[str]
    ) -> tuple[Type[BaseModel], str]:
        """Create schema for multiple array partial fields like ['script.type', 'script.video_heard']"""
        # Extract collection names and field names
        collection_fields = {}
        for spec in dot_notation_specs:
            collection_name, field_name = spec.split(".")
            if collection_name not in collection_fields:
                collection_fields[collection_name] = []
            collection_fields[collection_name].append(field_name)

        # Must all be from same collection
        if len(collection_fields) != 1:
            raise ValueError(
                f"Multi-field array partials must be from same collection: {dot_notation_specs}"
            )

        collection_name = list(collection_fields.keys())[0]
        field_names = collection_fields[collection_name]

        # Create schema with multiple fields
        schema = self.create_array_partial_schema_multi_field(
            collection_name, field_names
        )
        path = f"{collection_name}[].{'+'.join(field_names)}"
        return schema, path

    def _create_multi_scalar_schema(
        self, scalar_specs: List[str]
    ) -> tuple[Type[BaseModel], str]:
        """Create schema for multiple scalar fields like ['code', 'title']"""
        schema = self.main_model.__class__.create_partial_schema(scalar_specs)
        path = "+".join(scalar_specs)
        return schema, path

    def create_array_partial_schema_multi_field(
        self, collection_name: str, field_names: List[str]
    ) -> Type[BaseModel]:
        """
        Create a schema for array partial updates with multiple fields.

        Args:
            collection_name: e.g., "script"
            field_names: e.g., ["type", "video_heard"]

        Returns:
            Schema like {"script": [{"type": "...", "video_heard": "..."}]}
        """
        model_class = self._get_model_class_for_collection(collection_name)

        # Create partial schema with multiple fields
        partial_schema = model_class.create_partial_schema(field_names)

        # Wrap in array structure
        schema_name = f"ArrayPartial_{collection_name}_{'_'.join(field_names)}"
        schema = create_model(
            schema_name,
            **{collection_name: (List[partial_schema], ...)},
        )

        return schema

    def create_array_partial_schema(
        self, collection_name: str, field_name: str
    ) -> Type[BaseModel]:
        """
        Create schema for arrays of partial objects.

        Args:
            collection_name: e.g., "script"
            field_name: e.g., "video_heard"

        Returns:
            Schema like: {"script": [{"video_heard": str}]}
        """
        # Get the model class for this collection
        model_class = self._get_model_class_for_collection(collection_name)

        # Create partial model for just this field
        partial_model = model_class.create_partial_schema([field_name])

        # Create wrapper schema: {collection_name: [PartialModel]}
        wrapper_fields = {collection_name: (List[partial_model], [])}
        wrapper_schema = create_model(
            f"ArrayPartial_{collection_name}_{field_name}", **wrapper_fields
        )

        return wrapper_schema

    def apply_array_partial_response(
        self,
        ai_response: BaseModel,
        collection_name: str,
        field_name: str,
        append_mode: bool = False,
    ) -> str:
        """
        Apply AI response containing array of partial objects.

        Default behavior: OVERWRITE the entire array with new items (preserving other fields)
        Optional: Set append_mode=True to append instead of overwrite

        Args:
            ai_response: AI response like {"script": [{"video_heard": "..."}, {"video_heard": "..."}]}
            collection_name: e.g., "script"
            field_name: e.g., "video_heard"
            append_mode: If True, append new items. If False (default), overwrite array

        Returns:
            Path where data was applied
        """
        ai_data = ai_response.model_dump()

        if collection_name not in ai_data:
            raise ValueError(f"Expected '{collection_name}' in AI response")

        items_data = ai_data[collection_name]
        if not isinstance(items_data, list):
            raise ValueError(f"Expected array for '{collection_name}' in AI response")

        # Get current collection
        collection = getattr(self.main_model, collection_name)
        model_class = self._get_model_class_for_collection(collection_name)

        if append_mode:
            # APPEND MODE: Add new items to the end
            starting_index = len(collection)

            for item_data in items_data:
                partial_item = model_class(
                    **{field_name: item_data.get(field_name, "")}
                )
                collection.append(partial_item)

            path_range = f"{collection_name}[{starting_index}:{starting_index + len(items_data)}].{field_name}"
        else:
            # OVERWRITE MODE (default): Replace array contents while preserving other fields

            # First, preserve existing items' other fields if we're updating fewer items than exist
            preserved_items = []
            for i in range(min(len(collection), len(items_data))):
                current_item = collection[i]
                preserved_data = current_item.model_dump()
                # Update the target field with new value
                preserved_data[field_name] = items_data[i].get(field_name, "")
                preserved_items.append(model_class(**preserved_data))

            # Add any additional new items beyond existing array length
            for i in range(len(collection), len(items_data)):
                partial_item = model_class(
                    **{field_name: items_data[i].get(field_name, "")}
                )
                preserved_items.append(partial_item)

            # Replace the entire collection
            collection.clear()
            collection.extend(preserved_items)

            path_range = f"{collection_name}[0:{len(items_data)}].{field_name}"

        # Log the interaction
        self._log_interaction(path_range, ai_response)
        return path_range

    def append_ai_response_to_collection(
        self, ai_response: BaseModel, collection_name: str
    ) -> str:
        """
        Append a complete AI response object to a collection.
        Use this when the AI returns a complete object (like a full Doc with type and content)
        that you want to add to an array.

        Args:
            ai_response: The AI response (should be an instance of the collection's item type)
            collection_name: Name of the collection field (e.g., "docs", "items")

        Returns:
            The resolved field path where the data was applied
        """

        # Verify the collection exists
        if collection_name not in self.main_model.__class__.model_fields:
            raise ValueError(f"Collection '{collection_name}' not found in model")

        # Get the collection and append the new item
        collection = getattr(self.main_model, collection_name)
        next_index = len(collection)

        # Convert the AI response to the correct model type if needed
        model_class = self._get_model_class_for_collection(collection_name)
        if isinstance(ai_response, model_class):
            # Already the right type
            new_item = ai_response
        else:
            # Convert from AI response to the target model type
            new_item = model_class(**ai_response.model_dump())

        collection.append(new_item)

        resolved_path = f"{collection_name}.{next_index}"
        self._log_interaction(resolved_path, ai_response)

        return resolved_path

    def apply_ai_response_to_field_path(
        self, ai_response: BaseModel, field_path: str
    ) -> None:
        """
        Apply AI response to a field path, creating nested instances as needed.

        Args:
            ai_response: The AI's response (instance of partial schema)
            field_path: Dot notation path like "script.0.video_heard"
        """
        path_parts = field_path.split(".")

        if len(path_parts) == 1:
            # Root level field
            field_name = path_parts[0]
            self.main_model = self.main_model.merge_partial_update(
                ai_response, [field_name]
            )

        elif len(path_parts) == 3:
            # Nested field like "script.0.video_heard"
            collection_name, index_str, field_name = path_parts
            index = int(index_str)

            # Ensure the collection and item exist
            self._ensure_collection_item_exists(collection_name, index)

            # Get the target item and apply the update
            collection = getattr(self.main_model, collection_name)
            target_item = collection[index]
            updated_item = target_item.merge_partial_update(ai_response, [field_name])
            collection[index] = updated_item

        self._log_interaction(field_path, ai_response)

    def _ensure_collection_item_exists(self, collection_name: str, index: int) -> None:
        """Ensure a collection item exists at the given index."""
        collection = getattr(self.main_model, collection_name)

        # Extend collection if needed
        while len(collection) <= index:
            model_class = self._get_model_class_for_collection(collection_name)
            collection.append(model_class())

    def _get_all_nested_model_info(self) -> Dict[str, Type[UpdatableModel]]:
        """Get information about all nested model types."""
        nested_models = {}

        # You'll need to customize this for your specific models
        # This is the only part that needs to be adapted for different projects
        for field_name, field_info in self.main_model.__class__.model_fields.items():
            annotation = field_info.annotation

            # Handle Optional[List[...]] by extracting the inner type
            if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                # Check if this is Optional[List[...]] (Union[List[...], None])
                non_none_args = [
                    arg for arg in annotation.__args__ if arg is not type(None)
                ]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            # Check if it's a List[SomeModel] type
            if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                # Get the inner type (e.g., ScriptScene from List[ScriptScene])
                inner_type = annotation.__args__[0]
                if hasattr(
                    inner_type, "model_fields"
                ):  # Check if it's a Pydantic model
                    nested_models[field_name] = inner_type

        return nested_models

    def _get_model_class_for_collection(
        self, collection_name: str
    ) -> Type[UpdatableModel]:
        """Get the model class for a collection."""
        nested_models = self._get_all_nested_model_info()
        if collection_name in nested_models:
            return nested_models[collection_name]
        else:
            raise ValueError(f"Unknown collection: {collection_name}")

    def _log_interaction(self, field_path: str, ai_response: BaseModel) -> None:
        """Log the AI interaction."""
        self.interaction_history.append(
            {
                "field_path": field_path,
                "ai_response": ai_response.model_dump(),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        )
