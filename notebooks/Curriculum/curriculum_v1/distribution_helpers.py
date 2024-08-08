import aind_behavior_services.task_logic.distributions as distributions
import aind_behavior_vr_foraging.task_logic as vr_task_logic

def OperantLogicHelper(stop_duration: float = 0.5, is_operant: bool = False):
    """
    Helper function to create an instance of `vr_task_logic.OperantLogic`.

    Parameters:
    - stop_duration (float): The duration in seconds for which the agent should stop at each reward location. Default is 0.5.
    - is_operant (bool): Flag indicating whether the task is operant or not. Default is False.

    Returns:
    - vr_task_logic.OperantLogic: An instance of `vr_task_logic.OperantLogic` with the specified parameters.
    """
    return vr_task_logic.OperantLogic(
        is_operant=is_operant, 
        stop_duration=stop_duration, 
        time_to_collect_reward=100000, 
        grace_distance_threshold=5
    )

def NormalDistributionHelper(mean: 0.5, standard_deviation: 0.15, mininum: 0, maximum: 1000):
    """
    Helper function to create a normal distribution.

    Args:
        mean (float): The mean of the distribution.
        standard_deviation (float): The standard deviation of the distribution.
        mininum (int): The minimum value of the distribution.
        maximum (int): The maximum value of the distribution.

    Returns:
        distributions.NormalDistribution: The created normal distribution.
    """
    return distributions.NormalDistribution(
        distribution_parameters=distributions.NormalDistributionParameters(mean=mean, std=standard_deviation), 
        truncation_parameters=distributions.TruncationParameters(min=mininum, max=maximum, is_truncated=True), 
        scaling_parameters=distributions.ScalingParameters(scale=1.0, offset=0.0)
    )

def UniformDistributionHelper(minimum: 0, maximum: 1000):
    """
    Creates a uniform distribution helper.

    Args:
        minimum (int): The minimum value of the distribution.
        maximum (int): The maximum value of the distribution.

    Returns:
        distributions.UniformDistribution: The uniform distribution object.
    """
    return distributions.UniformDistribution(
        distribution_parameters=distributions.UniformDistributionParameters(min=minimum, max=maximum),
        truncation_parameters=distributions.TruncationParameters(min=0, max=10000, is_truncated=True),
        scaling_parameters=distributions.ScalingParameters(scale=1.0, offset=0.0)
    )

def ExponentialDistributionHelper(rate: float= 1, minimum: float =  0, maximum: float = 1000):
    """
    Creates an ExponentialDistribution object with the specified parameters.

    Parameters:
    - rate (float): The rate parameter of the exponential distribution. Default is 1.
    - minimum (float): The minimum value for truncation. Default is 0.
    - maximum (float): The maximum value for truncation. Default is 1000.

    Returns:
    - ExponentialDistribution: An ExponentialDistribution object with the specified parameters.
    """
    return distributions.ExponentialDistribution(
        distribution_parameters=distributions.ExponentialDistributionParameters(rate=rate),
        truncation_parameters=distributions.TruncationParameters(min=minimum, max=maximum, is_truncated=True),
        scaling_parameters=distributions.ScalingParameters(scale=1.0, offset=0.0),
    )

def Reward_VirtualSiteGeneratorHelper(rate: float = 0,contrast: float = 0.5, minimum= 25, maximum= 25, friction: float = 0):
    """
    Helper function to generate a virtual site generator for rewards.
    Args:
        rate (float, optional): The rate parameter for the exponential distribution. Defaults to 0.
        contrast (float, optional): The contrast value for the render specification. Defaults to 0.5.
        minimum (int, optional): The minimum value for the length distribution. Defaults to 25.
        maximum (int, optional): The maximum value for the length distribution. Defaults to 25.
        friction (float, optional): The friction value for the virtual site generator. Defaults to 0.
    Returns:
        vr_task_logic.VirtualSiteGenerator: The virtual site generator object for rewards.
    """
    if rate != 0:
        length_distribution = ExponentialDistributionHelper(rate = rate, minimum = minimum, maximum = maximum)
    else:
        length_distribution = UniformDistributionHelper(minimum, maximum)
        
    return vr_task_logic.VirtualSiteGenerator(
        render_specification=vr_task_logic.RenderSpecification(contrast=contrast),
        label=vr_task_logic.VirtualSiteLabels.REWARDSITE,
        length_distribution=length_distribution,
    )

def InterSite_VirtualSiteGeneratorHelper(rate: float = 0,contrast: float = 0.5, minimum= 40, maximum= 60, friction: float = 0):
    """
    Helper function to generate a virtual site generator for inter-site tasks.
    Parameters:
    - rate (float): The rate parameter for the exponential distribution of the length. Default is 0.
    - contrast (float): The contrast value for the render specification. Default is 0.5.
    - minimum (int): The minimum value for the length distribution. Default is 40.
    - maximum (int): The maximum value for the length distribution. Default is 60.
    - friction (float): The friction value for the virtual site generator. Default is 0.
    Returns:
    - vr_task_logic.VirtualSiteGenerator: The virtual site generator object.
    """
    if rate != 0:
        length_distribution = ExponentialDistributionHelper(rate = rate, minimum = minimum, maximum = maximum)
    else:
        length_distribution = UniformDistributionHelper(minimum, maximum)
        
    return vr_task_logic.VirtualSiteGenerator(
        render_specification=vr_task_logic.RenderSpecification(contrast=contrast),
        label=vr_task_logic.VirtualSiteLabels.INTERSITE,
        length_distribution=length_distribution,
    )

def InterPatch_VirtualSiteGeneratorHelper(rate: float = 0, contrast: float = 1, minimum= 100, maximum= 200, friction: float = 0):
    """
    Helper function to generate a virtual site generator for interpatch virtual sites.
    Parameters:
    - rate (float): The rate parameter for the length distribution. If rate is not equal to 0, an exponential distribution will be used. Default is 0.
    - contrast (float): The contrast value for the render specification. Default is 1.
    - minimum: The minimum value for the length distribution. Default is 100.
    - maximum: The maximum value for the length distribution. Default is 200.
    - friction (float): The friction value for the treadmill specification. Default is 0.
    Returns:
    - vr_task_logic.VirtualSiteGenerator: The virtual site generator object for interpatch virtual sites.
    """
    if rate != 0:
        length_distribution = ExponentialDistributionHelper(rate = rate, minimum = minimum, maximum = maximum)
    else:
        length_distribution = UniformDistributionHelper(minimum, maximum)
        
    return vr_task_logic.VirtualSiteGenerator(
        render_specification=vr_task_logic.RenderSpecification(contrast=contrast),
        label=vr_task_logic.VirtualSiteLabels.INTERPATCH,
        length_distribution=length_distribution,
        treadmill_specification= vr_task_logic.TreadmillSpecification(friction=vr_task_logic.scalar_value(friction))
    )

def PostPatch_VirtualSiteGeneratorHelper(rate: float = 0, minimum= 100, maximum= 200, contrast: float = 1, friction: float = 0.5):
    """
    Helper function to create a virtual site generator for post-patch VR tasks.
    Parameters:
    - rate (float): The rate parameter for the exponential distribution of site lengths. If rate is 0, a uniform distribution will be used instead.
    - minimum: The minimum length of the virtual sites.
    - maximum: The maximum length of the virtual sites.
    - contrast (float): The contrast value for rendering the virtual sites.
    - friction (float): The friction value for the virtual sites.
    Returns:
    - vr_task_logic.VirtualSiteGenerator: The virtual site generator object.
    """
    if rate != 0:
        length_distribution = ExponentialDistributionHelper(rate = rate, minimum = minimum, maximum = maximum)
    else:
        length_distribution = UniformDistributionHelper(minimum, maximum)
        
    return vr_task_logic.VirtualSiteGenerator(
        render_specification=vr_task_logic.RenderSpecification(contrast=contrast),
        label=vr_task_logic.VirtualSiteLabels.POSTPATCH,
        length_distribution=length_distribution,
    )
    
# This is how the reward functions work
# https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging/issues/196
def CountUntilDepleted(available_water: int = 21, probability_reward: float = 0.9, amount_drop: int = 5):
    """
    Calculates the reward specification for a VR task based on the given parameters.

    Parameters:
    - available_water (int): The initial amount of available water.
    - probability_reward (float): The probability of receiving a reward.
    - amount_drop (int): The amount of water dropped per reward.

    Returns:
    - vr_task_logic.RewardSpecification: The reward specification for the VR task.
    """
    return vr_task_logic.RewardSpecification(operant_logic=OperantLogicHelper(is_operant = False), 
                                        delay=NormalDistributionHelper(0.0, 0.15, 0.0,0.3), 
                                        reward_function= vr_task_logic.PatchRewardFunction(
                                            amount= vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=amount_drop)),
                                            probability=vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=0.9)),
                                            available=vr_task_logic.RewardFunction(vr_task_logic.LinearFunction(a=-amount_drop, b=available_water)),      
                                            depletion_rule=vr_task_logic.DepletionRule.ON_REWARD,),   
    )

def ExponentialRewardSize(amount_drop: int = 5):
    """
    Returns a reward specification for an exponential reward size.

    Parameters:
    - amount_drop (int): The amount to drop for the reward size.

    Returns:
    - vr_task_logic.RewardSpecification: The reward specification for the exponential reward size.
    """
    return vr_task_logic.RewardSpecification(operant_logic=OperantLogicHelper(is_operant = False), 
                                        delay=NormalDistributionHelper(0.5, 0.15, 0.25,0.75), 
                                        reward_function= vr_task_logic.PatchRewardFunction(
                                            amount= vr_task_logic.RewardFunction(vr_task_logic.PowerFunction(mininum=0, maximum= 10, a = amount_drop, b=2.718281828459045, c=-0.4)),
                                            probability=vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=0.9)),
                                            available=vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=1)),      
                                            depletion_rule=vr_task_logic.DepletionRule.ON_REWARD,),   
    )

def ExponentialProbabilityRewardCount(amount_drop: int = 5, available_water: int = 50):
    """
    Calculates the exponential probability reward count for a VR task.

    Parameters:
    - amount_drop (int): The amount of reward dropped.
    - available_water (int): The amount of available water.

    Returns:
    - vr_task_logic.RewardSpecification: The reward specification for the VR task.
    """
    return vr_task_logic.RewardSpecification(operant_logic=OperantLogicHelper(is_operant = False), 
                                        delay=NormalDistributionHelper(0.25, 0.15, 0.,0.75), 
                                        reward_function= vr_task_logic.PatchRewardFunction(
                                            amount= vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=amount_drop)),
                                            probability=vr_task_logic.RewardFunction(vr_task_logic.PowerFunction(mininum=0, maximum= 0.9, a = amount_drop, b=2.718281828459045, c=-0.025)),
                                            available=vr_task_logic.RewardFunction(vr_task_logic.LinearFunction(a=-amount_drop, b=available_water)),      
                                            depletion_rule=vr_task_logic.DepletionRule.ON_REWARD,),   
    )

def ExponentialProbabilityReward(amount_drop: int = 5, available_water: int = 50, c = -0.1, maximum_p = 0.9):
    """
    Calculates the exponential probability reward for a VR task.

    Parameters:
    - amount_drop (int): The amount of reward to be dropped.
    - available_water (int): The amount of available water.
    - c (float): The value of the constant 'c' in the power function.
    - maximum_p (float): The maximum value of the probability.

    Returns:
    - vr_task_logic.RewardSpecification: The reward specification for the VR task.
    """
    return vr_task_logic.RewardSpecification(operant_logic=OperantLogicHelper(is_operant = False), 
                                        delay=NormalDistributionHelper(0.25, 0.15, 0.,0.75), 
                                        reward_function= vr_task_logic.PatchRewardFunction(
                                            amount= vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=amount_drop)),
                                            probability=vr_task_logic.RewardFunction(vr_task_logic.PowerFunction(mininum=0, maximum= maximum_p, a = maximum_p, b=2.718281828459045, c=c)),
                                            available=vr_task_logic.RewardFunction(vr_task_logic.ConstantFunction(value=amount_drop)),      
                                            depletion_rule=vr_task_logic.DepletionRule.ON_REWARD,),
    )