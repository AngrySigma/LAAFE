import hydra
from omegaconf import DictConfig
from src.optimization.data_operations.dataset_loaders import load_dataset


@hydra.main(version_base=None,
            config_path='D:/PhD/LAAFE/src/optimization',
            config_name='cfg')
def main(cfg: DictConfig):
    dataset_id = cfg[cfg.problem_type].datasets[0]
    dataset = load_dataset(dataset_id)
    print(dataset)


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
