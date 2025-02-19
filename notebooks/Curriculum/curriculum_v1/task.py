"""
Example of Task creation
"""

import json
from typing import Literal

from pydantic import Field

import aind_behavior_vr_foraging.task_logic as vr_task_logic
import distribution_helpers as dist

class updaters:
    """
    Updaters
    """
    pass


vr_task_logic.AindVrForagingTaskLogic(
    name= 'test',
    )

operation_control = vr_task_logic.OperationControl(
    movable_spout_control=vr_task_logic.MovableSpoutControl(
        enabled=False,
        time_to_collect_after_reward=1,
        retracting_distance=2000,
    ),
    audio_control=vr_task_logic.AudioControl(duration=0.2, frequency=5000),
    odor_control=vr_task_logic.OdorControl(
        valve_max_open_time=10, target_odor_flow=100, target_total_flow=1000, use_channel_3_as_carrier=True
    ),
    position_control=vr_task_logic.PositionControl(
        gain=vr_task_logic.Vector3(x=1, y=1, z=1),
        initial_position=vr_task_logic.Vector3(x=0, y=2.56, z=0),
        frequency_filter_cutoff=5,
        velocity_threshold=8,
    ),
)

patch1 = vr_task_logic.PatchStatistics(
    label="Amyl Acetate",
    state_index=0,
    odor_specification=vr_task_logic.OdorSpecification(index=0, concentration=1),
    reward_specification=dist.CountUntilDepleted(available_water= 1000, amount_drop=5),
    virtual_site_generation=vr_task_logic.VirtualSiteGeneration(
        inter_patch=dist.InterPatch_VirtualSiteGeneratorHelper(),
        inter_site=dist.InterSite_VirtualSiteGeneratorHelper(),
        reward_site=dist.Reward_VirtualSiteGeneratorHelper(),
    ),
)

environment_statistics = vr_task_logic.EnvironmentStatistics(
    first_state=0, transition_matrix=vr_task_logic.Matrix2D(data=[[1]]), patches=[patch1]
)
    

task_logic=vr_task_logic.AindVrForagingTaskLogic(
    task_parameters=vr_task_logic.AindVrForagingTaskParameters(
        rng_seed=None,
        environment_statistics=environment_statistics,
        task_mode_settings=vr_task_logic.ForagingSettings(),
        operation_control=operation_control,
    )
)

with open(f'local/vr_task_preward_intercept_stage{stage}.json', "w") as f:
    f.write(task_logic.model_dump_json(indent=3))
    
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

    # # Optional: Add additional validation to fields.
    # @field_validator("field_1", "field_2")
    # @classmethod
    # def check_something(cls, v: int, info: ValidationInfo):
    #     """Your validation code here"""
    #     return v


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