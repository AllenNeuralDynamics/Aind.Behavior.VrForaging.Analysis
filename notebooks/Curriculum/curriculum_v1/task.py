"""
Example of Task creation
"""

import json
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from aind_behavior_curriculum import Task, TaskParameters


class ExampleTaskParameters(TaskParameters):
    """
    Example Task Parameters
    """

    # Required: Define type annotations for strict type checks.
    # Make fields immutable with Literal type.
    field_1: int = Field(default=0, ge=0.0)
    field_2: int = Field(default=0, ge=0.0)
    field_3: float = Field(default=0.5, ge=0.0, le=1.0)
    field_4: float = Field(default=0.5, ge=0.0, le=1.0)
    field_5: Literal["Immutable Field"] = "Immutable Field"

    # Optional: Add additional validation to fields.
    @field_validator("field_1", "field_2")
    @classmethod
    def check_something(cls, v: int, info: ValidationInfo):
        """Your validation code here"""
        return v


class ExampleTask(Task):
    """
    Example Task
    """

    name: Literal["TaskName"] = "TaskName"
    description: str = Field(default="Ex description of task")

    task_parameters: ExampleTaskParameters = Field(
        ..., description=ExampleTaskParameters.__doc__.strip()
    )


if __name__ == "__main__":
    # Create task, optionally add parameters
    ex_parameters = ExampleTaskParameters(field_2=50, field_4=0.8)
    ex_task = ExampleTask(task_parameters=ex_parameters)
    print(ex_task)

    # Update Task parameters individually
    ex_task.task_parameters.field_1 = 100
    ex_task.task_parameters.field_2 = 200
    print(ex_task)

    # Export/Serialize Task Schema:
    with open("examples/example_project/jsons/task_schema.json", "w") as f:
        json_dict = ExampleTask.model_json_schema()
        json_string = json.dumps(json_dict, indent=4)
        f.write(json_string)

    # Export/Serialize Instance:
    with open("examples/example_project/jsons/task_instance.json", "w") as f:
        json_dict = ex_task.model_dump()
        json_string = json.dumps(json_dict, indent=4)
        f.write(json_string)

    # Import/De-serialize Instance:
    with open("examples/example_project/jsons/task_instance.json", "r") as f:
        json_data = f.read()
    task_instance = ExampleTask.model_validate_json(json_data)
    print(task_instance)