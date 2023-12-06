import hydra
from omegaconf import DictConfig
from src.optimization.data_operations.dataset_loaders import DatasetLoader
from src.optimization.data_operations.operation_pipeline import \
    parse_data_operation_pipeline, apply_pipeline
from src.optimization.llm.llm_templates import LLMTemplate
from src.optimization.llm.gpt import ChatBot, ChatMessage, MessageHistory
from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostClassifier


@hydra.main(version_base=None,
            config_path='D:/PhD/LAAFE/',
            config_name='cfg')
def main(cfg: DictConfig):
    chatbot = ChatBot(api_key=cfg.llm.gpt.openai_api_key,
                      api_organization=cfg.llm.gpt.openai_api_organization,
                      model=cfg.llm.gpt.model_name)
    llm_template = LLMTemplate(operators=cfg.llm.operators)

    dataset_ids = cfg[cfg.problem_type].datasets[
                  :1]  # take first dataset for now
    dataset_loader = DatasetLoader(dataset_ids=dataset_ids)
    for dataset in dataset_loader:
        data_train, data_test, target_train, target_test = train_test_split(
            np.array(dataset.data), np.array(dataset.target), test_size=0.2)
        model = CatBoostClassifier()
        model.fit(data_train, target_train)
        metrics = model.score(data_test, target_test)

        messages = MessageHistory()
        messages.add_message(llm_template.initial_template())
        messages.add_message(str(dataset))
        messages.add_message(llm_template.previous_evaluations_template())
        messages.add_message(llm_template.instruction_template())
        messages.add_pipeline_evaluation('Initial evaluation', metrics)
        for _ in range(cfg.experiment.num_iterations):
            completion = chatbot.get_completion(messages=messages)
            try:
                # completion = 'Fillna_mean(V1), Fillna_median(V2), Fillna_mean(V3), Fillna_median(V4)'
                # completion = "Fillna_mean(duration), Fillna_mean(credit_amount), Fillna_mean(present_employment), Fillna_median(age), Minmax(duration), Minmax(credit_amount), Minmax(present_employment), Minmax(age)"
                pipeline = parse_data_operation_pipeline(completion)
                dataset_mod = apply_pipeline(dataset.data, pipeline)
                data_train, data_test, target_train, target_test = train_test_split(
                    np.array(dataset_mod), np.array(dataset.target),
                    test_size=0.2)
                # TODO: fix data leak
                # model = Fedot(problem=cfg.problem_type, preset='light',
                #               timeout=10)
                # model.fit(features=data_train, target=target_train)
                # prediction = model.predict(features=data_test)
                # metrics = model.get_metrics(target=target_test,
                #                             metric_names=cfg.metrics)
                model = CatBoostClassifier()
                model.fit(data_train, target_train)
                metrics = model.score(data_test, target_test)
                messages.add_pipeline_evaluation(completion, metrics)
            except Exception as e:
                with open(f'D:/PhD/LAAFE/results/results_{dataset.name}.txt',
                          'w') as f:
                    f.write(str(messages))
                    f.write(completion)
                    f.write(f'{type(e).__name__}, {str(e)}')
        with open(f'D:/PhD/LAAFE/results/results_{dataset.name}.txt',
                  'w') as f:
            f.write(str(messages))


if __name__ == '__main__':
    main()
