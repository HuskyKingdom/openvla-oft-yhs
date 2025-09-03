from energy_model.model import EnergyModel


energy_model = EnergyModel(vla.module.llm_dim,7,512,2,NUM_ACTIONS_CHUNK).to(device_id).to(torch.bfloat16)