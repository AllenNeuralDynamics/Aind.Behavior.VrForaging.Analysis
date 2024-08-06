"""
Example of Curriculum creation
"""

import json
from typing import Literal

from pydantic import Field

from aind_behavior_curriculum import (
    GRADUATED,
    INIT_STAGE,
    Curriculum,
    Metrics,
    Policy,
    PolicyTransition,
    Stage,
    StageGraph,
    StageTransition,
    Task,
    TaskParameters,
    get_task_types,
)



# --- TASKS ---
class TaskAParameters(TaskParameters):
    field_a: int = Field(default=0, validate_default=True)

class TaskA(Task):
    name: Literal["Stage A"] = "Stage A"
    task_parameters: TaskParameters = Field(
        ..., description="Fill w/ Parameter Defaults", validate_default=True
    )


class TaskBParameters(TaskParameters):
    field_b: float = Field(default=0.0)


class TaskB(Task):
    name: Literal["Stage B"] = "Stage B"
    task_parameters: TaskBParameters = Field(
        ..., description="Fill w/ Parameter Defaults"
    )

class TaskCParameters(TaskParameters):
    field_b: float = Field(default=0.0)


class TaskC(Task):
    name: Literal["Stage C"] = "Stage C"
    task_parameters: TaskCParameters = Field(
        ..., description="Fill w/ Parameter Defaults"
    )

# --- METRICS ---
class ExampleMetrics(Metrics):
    """
  Parameters
    ----------
    Pending
    """
    odor_sites: int = Field(default=0)
    rewarded_sites: int = Field(default=0)
    rewarded_sites_max: int = Field(default=0)
    visited_patches: int = Field(default=0)

# --- POLICIES ---
# ------------------ STAGE A policies ------------------

# ----------------Distances----------------
def stageA_policy_distanceA_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.intersite = 8
    task_params.reward = 8
    task_params.interpatch = 8
    return task_params

stageA_policy_distanceA = Policy(rule=stageA_policy_distanceA_rule)

def stageA_policy_distanceB_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.intersite = 8
    task_params.reward = 8
    task_params.interpatch = 8
    return task_params

stageA_policy_distanceB = Policy(rule=stageA_policy_distanceB_rule)

def stageA_policy_distanceC_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.intersite = 8
    task_params.reward = 8
    task_params.interpatch = 8
    return task_params

stageA_policy_distanceC = Policy(rule=stageA_policy_distanceC_rule)

def stageA_policy_distanceD_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.intersite = 8
    task_params.reward = 8
    task_params.interpatch = 8
    return task_params

stageA_policy_distanceD = Policy(rule=stageA_policy_distanceD_rule)

# ----------------Stops----------------
def stageA_policy_stopA_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.stop_duration_max = 0.5
    task_params.stop_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    return task_params

stageA_policy_stopA = Policy(rule=stageA_policy_stopA_rule)

def stageA_policy_stopB_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.stop_duration_max = 0.5
    task_params.stop_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    return task_params

stageA_policy_stopB = Policy(rule=stageA_policy_stopB_rule)

def stageA_policy_stopC_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.stop_duration_max = 0.5
    task_params.stop_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    return task_params

stageA_policy_stopC = Policy(rule=stageA_policy_stopC_rule)

def stageA_policy_stopD_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.stop_duration_max = 0.5
    task_params.stop_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    return task_params

stageA_policy_stopD = Policy(rule=stageA_policy_stopD_rule)

# ------------------ STAGE B policies ------------------
# ----------------Leaves----------------
def stageB_policy_leaveA_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.leave_duration_max = 0.5
    task_params.leave_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    
    return task_params

stageB_policy_leaveA = Policy(rule=stageB_policy_leaveA_rule)

def stageB_policy_leaveB_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.leave_duration_max = 0.5
    task_params.leave_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    
    return task_params

stageB_policy_leaveB = Policy(rule=stageB_policy_leaveB_rule)

def stageB_policy_leaveC_rule(
    metrics: ExampleMetrics, task_params: TaskParameters
) -> TaskParameters:
    task_params = task_params.model_copy(deep=True)
    task_params.leave_duration_max = 0.5
    task_params.leave_duration_min = 0.0
    task_params.velocity_max = 40
    task_params.velocity_min = 10
    task_params.delay_max = 0.2
    task_params.delay_min = 0.0
    
    return task_params

stageB_policy_leaveC = Policy(rule=stageB_policy_leaveC_rule)

# --- POLICY TRANSTITIONS ---
def odor_sites_200_rule(metrics: ExampleMetrics) -> bool:
    return metrics.odor_sites >= 200

odor_sites_200 = PolicyTransition(rule=odor_sites_200_rule)

def rewarded_sites_max_100_rule(metrics: ExampleMetrics) -> bool:
    return metrics.rewarded_sites_max >= 100

rewarded_sites_max_100 = PolicyTransition(rule=rewarded_sites_max_100_rule)

def visited_patches_25_rule(metrics: ExampleMetrics) -> bool:
    return metrics.visited_patches > 25

visited_patches_25 = PolicyTransition(rule=visited_patches_25_rule)

# --- STAGE TRANSITIONS ---
def rewarded_sites_max_100_rule(metrics: ExampleMetrics) -> bool:
    return metrics.visited_patches > 5

rewarded_sites_max_100_stage = StageTransition(rule=rewarded_sites_max_100_rule)

def visited_patches_25_stage_rule(metrics: ExampleMetrics) -> bool:
    return metrics.visited_patches > 25

visited_patches_25_stage = StageTransition(rule=visited_patches_25_stage_rule)

# --- CURRICULUM ---
Tasks = get_task_types()

class MyCurriculum(Curriculum):
    name: Literal["My Curriculum"] = "My Curriculum"
    # graph: StageGraph[Union[TaskA, TaskB, Graduated]] = Field(default=StageGraph())
    graph: StageGraph[Tasks] = Field(default=StageGraph[Tasks]())  # type: ignore


def construct_curriculum() -> MyCurriculum:
    """
    Useful for testing.
    """

    with open("examples/example_project/jsons/schema.json", "w") as f:
        f.write(json.dumps(MyCurriculum.model_json_schema(), indent=4))

    # Init Stages
    taskA = TaskA(task_parameters=TaskAParameters())
    taskB = TaskB(task_parameters=TaskBParameters())
    taskC = TaskC(task_parameters=TaskCParameters())
    
    stageA = Stage(name="StageA", task=taskA)
    stageB = Stage(name="StageB", task=taskB)
    stageC = Stage(name="StageC", task=taskC)
    
    stageA.add_policy_transition(stageA_policy_distanceA, stageA_policy_distanceB_rule, odor_sites_200)
    stageA.add_policy_transition(stageA_policy_distanceB, stageA_policy_distanceC_rule, odor_sites_200)
    stageA.add_policy_transition(stageA_policy_distanceC, stageA_policy_distanceD_rule, odor_sites_200)
    
    stageA.add_policy_transition(stageA_policy_stopA, stageA_policy_stopB_rule, rewarded_sites_max_100)
    stageA.add_policy_transition(stageA_policy_stopB, stageA_policy_stopC_rule, rewarded_sites_max_100)
    stageA.add_policy_transition(stageA_policy_stopC, stageA_policy_stopD_rule, rewarded_sites_max_100)
    
    stageA.set_start_policies([stageA_policy_distanceA, stageA_policy_stopA_rule])

    stageB.add_policy_transition(stageB_policy_leaveA, stageB_policy_leaveB, visited_patches_25)
    stageB.add_policy_transition(stageB_policy_leaveB, stageB_policy_leaveC, visited_patches_25)
    stageB.set_start_policies(stageB_policy_leaveA_rule)

    # Construct the Curriculum
    curr = MyCurriculum(name="My Curriculum")
    curr.add_stage_transition(stageA, stageB, t2_5)
    curr.add_stage_transition(stageB, stageC, t2_10)

    return curr


if __name__ == "__main__":
    curr = construct_curriculum()

    with open("jsons/stage_instance.json", "w") as f:
        stageA = curr.see_stages()[0]
        json_dict = stageA.model_dump()
        json_string = json.dumps(json_dict, indent=4)
        f.write(json_string)

    with open("jsons/curr_instance.json", "w") as f:
        json_dict = curr.model_dump()
        json_string = json.dumps(json_dict, indent=4)
        f.write(json_string)

    curr.export_diagram(
        "diagrams/vr_foraging_diagram.png"
    )