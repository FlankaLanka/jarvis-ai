"""
Jarvis Voice Assistant - Data Validator Service

Validates data from external sources (GitHub, APIs) before use.
"""

from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ValidationStatus(Enum):
    """Status of data validation."""
    VALID = "valid"
    INVALID = "invalid"
    STALE = "stale"
    PARTIAL = "partial"


@dataclass
class ValidationResult:
    """Result of data validation."""
    status: ValidationStatus
    errors: List[str]
    warnings: List[str]
    data: Optional[Any] = None


class DataValidator:
    """
    Service for validating data from external sources.
    
    Provides:
    1. Schema validation
    2. Freshness checks
    3. Type validation
    4. Required field validation
    """
    
    def __init__(self, max_data_age_minutes: int = 5):
        """
        Initialize the validator.
        
        Args:
            max_data_age_minutes: Maximum age of data before considered stale
        """
        self._max_age = timedelta(minutes=max_data_age_minutes)
    
    def validate(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None,
        fetched_at: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Validate data against optional schema.
        
        Args:
            data: The data to validate
            schema: Optional schema definition
            fetched_at: When the data was fetched (for freshness)
            
        Returns:
            ValidationResult with status and any errors
        """
        errors = []
        warnings = []
        
        # Check for null data
        if data is None:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                errors=["Data is null"],
                warnings=[],
                data=None
            )
        
        # Check freshness
        if fetched_at:
            age = datetime.utcnow() - fetched_at
            if age > self._max_age:
                warnings.append(f"Data is {age.seconds // 60} minutes old")
        
        # Validate against schema if provided
        if schema:
            schema_errors = self._validate_schema(data, schema)
            errors.extend(schema_errors)
        
        # Determine status
        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.STALE if "old" in str(warnings) else ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            errors=errors,
            warnings=warnings,
            data=data if status != ValidationStatus.INVALID else None
        )
    
    def _validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Validate data against a schema definition.
        
        Args:
            data: The data to validate
            schema: Schema definition
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check type
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(data, expected_type):
                errors.append(f"Expected type {expected_type}, got {type(data).__name__}")
                return errors
        
        # For dictionaries, check required fields
        if isinstance(data, dict):
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate nested properties
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in data:
                    nested_errors = self._validate_schema(data[key], prop_schema)
                    errors.extend([f"{key}.{e}" for e in nested_errors])
        
        # For lists, validate items
        if isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    item_errors = self._validate_schema(item, items_schema)
                    errors.extend([f"[{i}].{e}" for e in item_errors])
        
        return errors
    
    def _check_type(self, data: Any, expected_type: str) -> bool:
        """
        Check if data matches expected type.
        
        Args:
            data: The data to check
            expected_type: Expected type name
            
        Returns:
            True if type matches
        """
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        
        expected = type_mapping.get(expected_type)
        if expected is None:
            return True  # Unknown type, pass
        
        return isinstance(data, expected)
    
    def validate_github_response(
        self,
        data: Any,
        expected_type: str = "file"
    ) -> ValidationResult:
        """
        Validate a GitHub API response.
        
        Args:
            data: The response data
            expected_type: Expected response type (file, search, commits)
            
        Returns:
            ValidationResult
        """
        schemas = {
            "file": {
                "type": "object",
                "required": ["path", "content"],
            },
            "search": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["path", "repository"],
                }
            },
            "commits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["sha", "message"],
                }
            },
        }
        
        schema = schemas.get(expected_type, {})
        return self.validate(data, schema)
    
    def validate_api_response(
        self,
        data: Any,
        schema: Dict[str, Any],
        fetched_at: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Validate a public API response.
        
        Args:
            data: The response data
            schema: Expected schema
            fetched_at: When the data was fetched
            
        Returns:
            ValidationResult
        """
        return self.validate(data, schema, fetched_at)
    
    def is_fresh(self, fetched_at: datetime) -> bool:
        """
        Check if data is still fresh.
        
        Args:
            fetched_at: When the data was fetched
            
        Returns:
            True if data is still fresh
        """
        return datetime.utcnow() - fetched_at <= self._max_age


# Global validator instance
data_validator = DataValidator()

