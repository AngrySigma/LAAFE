import hydra
from omegaconf import DictConfig
from src.optimization.data_operations.dataset_loaders import DatasetLoader
from src.optimization.llm.llm_templates import LLMTemplate
from src.optimization.llm.gpt import ChatBot, ChatMessage


@hydra.main(version_base=None,
            config_path='D:/PhD/LAAFE/',
            config_name='cfg')
def main(cfg: DictConfig):
    dataset_id = cfg[cfg.problem_type].datasets[0]
    dataset_loader = DatasetLoader(dataset_ids=[dataset_id])
    dataset = dataset_loader[0]
    llm_template = LLMTemplate(operators=cfg.llm.operators)
    chatbot = ChatBot(api_key=cfg.llm.gpt.openai_api_key,
                      api_organization=cfg.llm.gpt.openai_api_organization,
                      model=cfg.llm.gpt.model_name)
    print(chatbot.get_completion(messages=[ChatMessage(role='user', content='hello')]))
    # print(llm_template)
    # print(dataset)


# define data operation pipeline format (fix)


# pass to LLM

# get metrics
# get models


# inject data into LLM (maybe different, so it should get description from appropriate dataset loader)
# inject features into LLM
# inject target into LLM
# inject metrics into LLM
# inject models into LLM

# run LLM to get data operation pipeline
# # rerun if error encountered
# parse data operation pipeline
# run data operation pipeline
# train model
# get metrics
# return metrics if max iterations or target reached, otherwise return to LLM

# write final metrics to variable
# write final data operation pipeline to variable
# repeat for all datasets
# write final metrics to file

if __name__ == '__main__':
    main()
