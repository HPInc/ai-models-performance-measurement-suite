import argparse
import json
import sys

def resolve_ref(schema, root):
    """Recursively resolves a $ref in the schema within the root document."""
    if not isinstance(schema, dict):
        return schema
    
    ref = schema.get('$ref')
    if not ref:
        return schema
        
    if ref.startswith('#/'):
        path_parts = ref.split('/')[1:]
        current_node = root
        try:
            for part in path_parts:
                current_node = current_node[part]
            return resolve_ref(current_node, root)
        except (KeyError, TypeError):
            return schema
            
    return schema

def to_jsonc(data, schema, root, indent=0):
    """Recursively builds the JSONC string with description comments."""
    indent_str = "  " * indent
    next_indent_str = "  " * (indent + 1)
    
    # Resolve the current schema to handle generic refs
    effective_schema = resolve_ref(schema, root)
    
    # Handle 'anyOf' by finding the best matching sub-schema
    if isinstance(effective_schema, dict) and 'anyOf' in effective_schema:
        for option in effective_schema['anyOf']:
            resolved_option = resolve_ref(option, root)
            # Basic type matching heuristic
            if isinstance(data, dict) and (resolved_option.get('type') == 'object' or 'properties' in resolved_option):
                effective_schema = resolved_option
                break
            if isinstance(data, list) and (resolved_option.get('type') == 'array' or 'items' in resolved_option):
                effective_schema = resolved_option
                break
            if data is None and resolved_option.get('type') == 'null':
                return "null"

    if isinstance(data, dict):
        if not data:
            return "{}"
            
        lines = ["{"]
        keys = list(data.keys())
        
        for i, key in enumerate(keys):
            value = data[key]
            
            # Find the property schema
            prop_schema = effective_schema.get('properties', {}).get(key, {})
            
            # Look for description on the property definition itself
            description = prop_schema.get('description')
            
            # If not found, resolve ref and check again
            if not description:
                resolved_prop = resolve_ref(prop_schema, root)
                description = resolved_prop.get('description')
            
            # Add description comment
            if description:
                lines.append(f"{next_indent_str}// {description}")
                
            # Recursive call
            val_str = to_jsonc(value, prop_schema, root, indent + 1)
            
            comma = "," if i < len(keys) - 1 else ""
            lines.append(f'{next_indent_str}"{key}": {val_str}{comma}')
            
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    elif isinstance(data, list):
        if not data:
            return "[]"
            
        lines = ["["]
        items_schema = effective_schema.get('items', {})
        
        for i, item in enumerate(data):
            val_str = to_jsonc(item, items_schema, root, indent + 1)
            comma = "," if i < len(data) - 1 else ""
            lines.append(f"{next_indent_str}{val_str}{comma}")
            
        lines.append(f"{indent_str}]")
        return "\n".join(lines)
        
    elif data is None:
        return "null"
        
    else:
        return json.dumps(data)

def main():
    parser = argparse.ArgumentParser(description="Enrich JSON with comments from JSON schema.")
    parser.add_argument("input_path", help="Path to input JSON data file")
    parser.add_argument("schema_path", help="Path to JSON schema file")
    parser.add_argument("output_path", help="Path to output JSONC file")
    
    args = parser.parse_args()
    
    try:
        with open(args.input_path, 'r') as f:
            data = json.load(f)
        
        with open(args.schema_path, 'r') as f:
            root_schema = json.load(f)
            
        jsonc_output = to_jsonc(data, root_schema, root_schema)
        
        with open(args.output_path, 'w') as f:
            f.write(jsonc_output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
