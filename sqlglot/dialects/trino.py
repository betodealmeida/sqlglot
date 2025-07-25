from __future__ import annotations

from sqlglot import exp, parser, transforms
from sqlglot.dialects.dialect import (
    merge_without_target_sql,
    trim_sql,
    timestrtotime_sql,
    groupconcat_sql,
)
from sqlglot.dialects.presto import amend_exploded_column_table, Presto
from sqlglot.tokens import TokenType
import typing as t


class Trino(Presto):
    SUPPORTS_USER_DEFINED_TYPES = False
    LOG_BASE_FIRST = True

    class Parser(Presto.Parser):
        FUNCTION_PARSERS = {
            **Presto.Parser.FUNCTION_PARSERS,
            "TRIM": lambda self: self._parse_trim(),
            "JSON_QUERY": lambda self: self._parse_json_query(),
            "LISTAGG": lambda self: self._parse_string_agg(),
        }

        JSON_QUERY_OPTIONS: parser.OPTIONS_TYPE = {
            **dict.fromkeys(
                ("WITH", "WITHOUT"),
                (
                    ("WRAPPER"),
                    ("ARRAY", "WRAPPER"),
                    ("CONDITIONAL", "WRAPPER"),
                    ("CONDITIONAL", "ARRAY", "WRAPPED"),
                    ("UNCONDITIONAL", "WRAPPER"),
                    ("UNCONDITIONAL", "ARRAY", "WRAPPER"),
                ),
            ),
        }

        def _parse_json_query_quote(self) -> t.Optional[exp.JSONExtractQuote]:
            if not (
                self._match_text_seq("KEEP", "QUOTES") or self._match_text_seq("OMIT", "QUOTES")
            ):
                return None

            return self.expression(
                exp.JSONExtractQuote,
                option=self._tokens[self._index - 2].text.upper(),
                scalar=self._match_text_seq("ON", "SCALAR", "STRING"),
            )

        def _parse_json_query(self) -> exp.JSONExtract:
            return self.expression(
                exp.JSONExtract,
                this=self._parse_bitwise(),
                expression=self._match(TokenType.COMMA) and self._parse_bitwise(),
                option=self._parse_var_from_options(self.JSON_QUERY_OPTIONS, raise_unmatched=False),
                json_query=True,
                quote=self._parse_json_query_quote(),
                on_condition=self._parse_on_condition(),
            )
        
        def _parse_cte(self) -> t.Optional[exp.CTE]:
            # Handle inline UDF syntax: WITH FUNCTION name(...) RETURNS type RETURN expr
            if self._match(TokenType.FUNCTION):
                # Parse function name and parameters
                func = self._parse_user_defined_function(kind=TokenType.FUNCTION)
                if not func:
                    self.raise_error("Expected function name")
                    return None
                
                # Parse RETURNS type
                if not self._match_text_seq("RETURNS"):
                    self.raise_error("Expected RETURNS keyword")
                    return None
                
                returns_type = self._parse_type()
                
                # Parse function body
                if not self._match_texts(("RETURN", "BEGIN")):
                    self.raise_error("Expected RETURN or BEGIN keyword")
                    return None
                
                is_begin = self._prev.text.upper() == "BEGIN"
                body = self._parse_user_defined_function_expression()
                
                if is_begin:
                    self._match_text_seq("END")
                
                # Create a CTE-like structure for the inline function
                # We represent it as a CTE with a CREATE FUNCTION statement
                create_func = self.expression(
                    exp.Create,
                    this=func,
                    kind="FUNCTION",
                    expression=body,
                    properties=self.expression(
                        exp.Properties,
                        expressions=[
                            self.expression(
                                exp.Property,
                                this=exp.Literal.string("returns"),
                                value=returns_type
                            )
                        ]
                    ),
                    exists=False
                )
                
                # Wrap in CTE with function name as alias
                return self.expression(
                    exp.CTE,
                    this=create_func,
                    alias=self.expression(exp.TableAlias, this=func.this)
                )
            
            # Otherwise, use the parent CTE parser
            return super()._parse_cte()

    class Generator(Presto.Generator):
        PROPERTIES_LOCATION = {
            **Presto.Generator.PROPERTIES_LOCATION,
            exp.LocationProperty: exp.Properties.Location.POST_WITH,
        }
        
        def cte_sql(self, expression: exp.CTE) -> str:
            # Check if this is an inline function CTE
            if isinstance(expression.this, exp.Create) and expression.this.args.get("kind") == "FUNCTION":
                create = expression.this
                func = create.this
                
                # Extract function name
                func_name = self.sql(func.this)
                
                # Extract parameters
                params = []
                for param in func.expressions:
                    if isinstance(param, exp.ColumnDef):
                        param_name = self.sql(param.this)
                        param_type = self.sql(param.kind)
                        params.append(f"{param_name} {param_type}")
                params_str = ", ".join(params)
                
                # Extract return type
                returns_prop = None
                if hasattr(create, 'args') and 'properties' in create.args and create.args['properties']:
                    for prop in create.args['properties'].expressions:
                        if isinstance(prop, exp.Property) and prop.this.this == "returns":
                            returns_prop = prop.args.get('value')
                            break
                
                returns_type = self.sql(returns_prop) if returns_prop else "INTEGER"
                
                # Extract body
                body = self.sql(create.expression)
                
                # Generate inline function syntax
                return f"FUNCTION {func_name}({params_str}) RETURNS {returns_type} RETURN {body}"
            
            # Otherwise use parent implementation
            return super().cte_sql(expression)

        TRANSFORMS = {
            **Presto.Generator.TRANSFORMS,
            exp.ArraySum: lambda self,
            e: f"REDUCE({self.sql(e, 'this')}, 0, (acc, x) -> acc + x, acc -> acc)",
            exp.ArrayUniqueAgg: lambda self, e: f"ARRAY_AGG(DISTINCT {self.sql(e, 'this')})",
            exp.GroupConcat: lambda self, e: groupconcat_sql(self, e, on_overflow=True),
            exp.LocationProperty: lambda self, e: self.property_sql(e),
            exp.Merge: merge_without_target_sql,
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_qualify,
                    transforms.eliminate_distinct_on,
                    transforms.explode_projection_to_unnest(1),
                    transforms.eliminate_semi_and_anti_joins,
                    amend_exploded_column_table,
                ]
            ),
            exp.TimeStrToTime: lambda self, e: timestrtotime_sql(self, e, include_precision=True),
            exp.Trim: trim_sql,
        }

        SUPPORTED_JSON_PATH_PARTS = {
            exp.JSONPathKey,
            exp.JSONPathRoot,
            exp.JSONPathSubscript,
        }

        def jsonextract_sql(self, expression: exp.JSONExtract) -> str:
            if not expression.args.get("json_query"):
                return super().jsonextract_sql(expression)

            json_path = self.sql(expression, "expression")
            option = self.sql(expression, "option")
            option = f" {option}" if option else ""

            quote = self.sql(expression, "quote")
            quote = f" {quote}" if quote else ""

            on_condition = self.sql(expression, "on_condition")
            on_condition = f" {on_condition}" if on_condition else ""

            return self.func(
                "JSON_QUERY",
                expression.this,
                json_path + option + quote + on_condition,
            )
