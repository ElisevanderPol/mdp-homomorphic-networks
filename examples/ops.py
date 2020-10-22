def get_agent_cls_cartpole(agent_type, algo="ppo"):
    """
    Get agent wrapper for cartpole
    """
    if agent_type in ["equivariant", "nullspace", "random"]:
        from rlpyt.agents.pg.cartpole import CartpoleBasisAgent
        return CartpoleBasisAgent, agent_type
    elif agent_type == "mlp":
        from rlpyt.agents.pg.cartpole import CartpoleFfAgent
        return CartpoleFfAgent, None
    else:
        raise TypeError("No agent of type {agent_type} known")
