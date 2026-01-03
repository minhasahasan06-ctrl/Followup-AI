"""
CohortDSL Service (Phase C.6-C.7)
=================================
Domain-Specific Language for cohort definition and safe SQL compilation.

Tasks:
- C.6: CohortDSL Pydantic model
- C.7: Safe SQL compiler (parameterized queries only)
"""

import hashlib
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ComparisonOperator(str, Enum):
    """Safe comparison operators for DSL"""
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    LIKE = "like"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class LogicalOperator(str, Enum):
    """Logical operators for combining filters"""
    AND = "and"
    OR = "or"


class AggregateFunction(str, Enum):
    """Safe aggregate functions"""
    COUNT = "count"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


ALLOWED_TABLES = {
    "patients",
    "patient_vitals",
    "patient_conditions",
    "patient_medications",
    "symptom_checkins",
    "risk_assessments",
}

ALLOWED_COLUMNS = {
    "patients": {"id", "age", "gender", "created_at", "is_active"},
    "patient_vitals": {"patient_id", "vital_type", "value", "recorded_at"},
    "patient_conditions": {"patient_id", "condition_code", "status", "diagnosed_at"},
    "patient_medications": {"patient_id", "medication_code", "status", "started_at"},
    "symptom_checkins": {"patient_id", "symptom_type", "severity", "recorded_at"},
    "risk_assessments": {"patient_id", "risk_score", "risk_level", "assessed_at"},
}


class CohortFilter(BaseModel):
    """Single filter condition in DSL"""
    table: str = Field(..., description="Table name")
    column: str = Field(..., description="Column name")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    value: Optional[Union[str, int, float, bool, List[Any]]] = Field(None, description="Filter value")
    value_end: Optional[Union[str, int, float]] = Field(None, description="End value for BETWEEN")
    
    @field_validator("table")
    @classmethod
    def validate_table(cls, v: str) -> str:
        if v not in ALLOWED_TABLES:
            raise ValueError(f"Table '{v}' not in allowed tables: {ALLOWED_TABLES}")
        return v
    
    @field_validator("column")
    @classmethod
    def validate_column(cls, v: str, info) -> str:
        if not re.match(r"^[a-z_][a-z0-9_]*$", v):
            raise ValueError(f"Invalid column name: {v}")
        return v


class CohortFilterGroup(BaseModel):
    """Group of filters combined with logical operator"""
    logical_op: LogicalOperator = LogicalOperator.AND
    filters: List[Union[CohortFilter, "CohortFilterGroup"]] = Field(default_factory=list)


CohortFilterGroup.model_rebuild()


class CohortAggregate(BaseModel):
    """Aggregate definition for cohort statistics"""
    function: AggregateFunction
    table: str
    column: str
    alias: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    
    @field_validator("table")
    @classmethod
    def validate_table(cls, v: str) -> str:
        if v not in ALLOWED_TABLES:
            raise ValueError(f"Table '{v}' not in allowed tables")
        return v


class CohortDSL(BaseModel):
    """
    C.6: CohortDSL Pydantic model for cohort definitions.
    
    Represents a structured, validated cohort query that can be
    safely compiled to SQL with parameterized queries.
    """
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    base_table: str = Field(default="patients")
    filter_group: CohortFilterGroup = Field(default_factory=CohortFilterGroup)
    aggregates: List[CohortAggregate] = Field(default_factory=list)
    limit: Optional[int] = Field(default=None, ge=1, le=10000)
    schema_version: str = Field(default="1.0")
    
    @field_validator("base_table")
    @classmethod
    def validate_base_table(cls, v: str) -> str:
        if v not in ALLOWED_TABLES:
            raise ValueError(f"Base table '{v}' not allowed")
        return v
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of DSL for reproducibility tracking"""
        import json
        dsl_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(dsl_str.encode('utf-8')).hexdigest()


class CohortSQLCompiler:
    """
    C.7: Safe SQL compiler for CohortDSL.
    
    SECURITY: Only produces parameterized queries, never string interpolation.
    All values are bound as parameters to prevent SQL injection.
    """
    
    def __init__(self):
        self._param_counter = 0
        self._params: Dict[str, Any] = {}
    
    def _next_param(self, value: Any) -> str:
        """Generate next parameter name and store value"""
        self._param_counter += 1
        param_name = f"p{self._param_counter}"
        self._params[param_name] = value
        return param_name
    
    def _validate_column_for_table(self, table: str, column: str) -> bool:
        """Validate column exists in table's allowed columns"""
        allowed = ALLOWED_COLUMNS.get(table, set())
        return column in allowed
    
    def _compile_filter(self, f: CohortFilter, table_alias: str = "") -> str:
        """Compile single filter to SQL clause with parameterized value"""
        if not self._validate_column_for_table(f.table, f.column):
            raise ValueError(f"Column '{f.column}' not allowed for table '{f.table}'")
        
        col_ref = f"{table_alias}.{f.column}" if table_alias else f"{f.table}.{f.column}"
        
        if f.operator == ComparisonOperator.IS_NULL:
            return f"{col_ref} IS NULL"
        elif f.operator == ComparisonOperator.IS_NOT_NULL:
            return f"{col_ref} IS NOT NULL"
        elif f.operator == ComparisonOperator.EQ:
            param = self._next_param(f.value)
            return f"{col_ref} = :{param}"
        elif f.operator == ComparisonOperator.NE:
            param = self._next_param(f.value)
            return f"{col_ref} != :{param}"
        elif f.operator == ComparisonOperator.GT:
            param = self._next_param(f.value)
            return f"{col_ref} > :{param}"
        elif f.operator == ComparisonOperator.GTE:
            param = self._next_param(f.value)
            return f"{col_ref} >= :{param}"
        elif f.operator == ComparisonOperator.LT:
            param = self._next_param(f.value)
            return f"{col_ref} < :{param}"
        elif f.operator == ComparisonOperator.LTE:
            param = self._next_param(f.value)
            return f"{col_ref} <= :{param}"
        elif f.operator == ComparisonOperator.IN:
            if not isinstance(f.value, list):
                raise ValueError("IN operator requires list value")
            params = [self._next_param(v) for v in f.value]
            param_refs = ", ".join(f":{p}" for p in params)
            return f"{col_ref} IN ({param_refs})"
        elif f.operator == ComparisonOperator.NOT_IN:
            if not isinstance(f.value, list):
                raise ValueError("NOT_IN operator requires list value")
            params = [self._next_param(v) for v in f.value]
            param_refs = ", ".join(f":{p}" for p in params)
            return f"{col_ref} NOT IN ({param_refs})"
        elif f.operator == ComparisonOperator.BETWEEN:
            param1 = self._next_param(f.value)
            param2 = self._next_param(f.value_end)
            return f"{col_ref} BETWEEN :{param1} AND :{param2}"
        elif f.operator == ComparisonOperator.LIKE:
            param = self._next_param(f.value)
            return f"{col_ref} LIKE :{param}"
        else:
            raise ValueError(f"Unknown operator: {f.operator}")
    
    def _compile_filter_group(self, group: CohortFilterGroup, table_alias: str = "") -> str:
        """Compile filter group to SQL with proper logical operators"""
        if not group.filters:
            return "1=1"
        
        clauses = []
        for f in group.filters:
            if isinstance(f, CohortFilter):
                clauses.append(self._compile_filter(f, table_alias))
            elif isinstance(f, CohortFilterGroup):
                sub_clause = self._compile_filter_group(f, table_alias)
                clauses.append(f"({sub_clause})")
        
        op = " AND " if group.logical_op == LogicalOperator.AND else " OR "
        return op.join(clauses)
    
    def compile_count_query(self, dsl: CohortDSL) -> Tuple[str, Dict[str, Any]]:
        """
        Compile DSL to COUNT query with parameters.
        
        Returns: (sql_string, params_dict)
        """
        self._param_counter = 0
        self._params = {}
        
        where_clause = self._compile_filter_group(dsl.filter_group)
        
        sql = f"""
            SELECT COUNT(DISTINCT {dsl.base_table}.id) as patient_count
            FROM {dsl.base_table}
            WHERE {where_clause}
        """
        
        return sql.strip(), self._params.copy()
    
    def compile_aggregate_query(self, dsl: CohortDSL) -> Tuple[str, Dict[str, Any]]:
        """
        Compile DSL to aggregate query with parameters.
        
        Returns: (sql_string, params_dict)
        """
        self._param_counter = 0
        self._params = {}
        
        if not dsl.aggregates:
            return self.compile_count_query(dsl)
        
        where_clause = self._compile_filter_group(dsl.filter_group)
        
        agg_parts = []
        for agg in dsl.aggregates:
            if not self._validate_column_for_table(agg.table, agg.column):
                raise ValueError(f"Column '{agg.column}' not allowed for table '{agg.table}'")
            
            col_ref = f"{agg.table}.{agg.column}"
            
            if agg.function == AggregateFunction.COUNT:
                agg_parts.append(f"COUNT({col_ref}) as {agg.alias}")
            elif agg.function == AggregateFunction.AVG:
                agg_parts.append(f"AVG({col_ref}) as {agg.alias}")
            elif agg.function == AggregateFunction.SUM:
                agg_parts.append(f"SUM({col_ref}) as {agg.alias}")
            elif agg.function == AggregateFunction.MIN:
                agg_parts.append(f"MIN({col_ref}) as {agg.alias}")
            elif agg.function == AggregateFunction.MAX:
                agg_parts.append(f"MAX({col_ref}) as {agg.alias}")
        
        agg_select = ", ".join(agg_parts)
        
        sql = f"""
            SELECT COUNT(DISTINCT {dsl.base_table}.id) as patient_count,
                   {agg_select}
            FROM {dsl.base_table}
            WHERE {where_clause}
        """
        
        return sql.strip(), self._params.copy()
    
    def compile_ids_query(self, dsl: CohortDSL) -> Tuple[str, Dict[str, Any]]:
        """
        Compile DSL to query returning patient IDs.
        
        NOTE: For internal use only - IDs should never be exposed externally.
        
        Returns: (sql_string, params_dict)
        """
        self._param_counter = 0
        self._params = {}
        
        where_clause = self._compile_filter_group(dsl.filter_group)
        
        limit_clause = ""
        if dsl.limit:
            limit_clause = f"LIMIT {dsl.limit}"
        
        sql = f"""
            SELECT DISTINCT {dsl.base_table}.id
            FROM {dsl.base_table}
            WHERE {where_clause}
            {limit_clause}
        """
        
        return sql.strip(), self._params.copy()


def compile_cohort_dsl(dsl: CohortDSL) -> Tuple[str, Dict[str, Any]]:
    """Convenience function to compile CohortDSL to count query"""
    compiler = CohortSQLCompiler()
    return compiler.compile_count_query(dsl)


def compile_cohort_aggregates(dsl: CohortDSL) -> Tuple[str, Dict[str, Any]]:
    """Convenience function to compile CohortDSL to aggregate query"""
    compiler = CohortSQLCompiler()
    return compiler.compile_aggregate_query(dsl)


__all__ = [
    "CohortDSL",
    "CohortFilter",
    "CohortFilterGroup",
    "CohortAggregate",
    "ComparisonOperator",
    "LogicalOperator",
    "AggregateFunction",
    "CohortSQLCompiler",
    "compile_cohort_dsl",
    "compile_cohort_aggregates",
    "ALLOWED_TABLES",
    "ALLOWED_COLUMNS",
]
