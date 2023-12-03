import hydra
from src.optimization.data_operations.operation_aliases import OPERATIONS


class LLMTemplate:
    def __init__(self, operators, experiment_description=None, output_format=None, instruction=None):
        self.experiment_description = experiment_description if (
                experiment_description is not None) else (
            'You should perform a feature engineering for the provided dataset to improve the performance of the model. ')
        self.output_format = output_format if (
                output_format is not None) else (
            'The output in an operation pipeline in following format:'
            '\n"operation1(input1, input2, ...) -> operation2(input1, input2, ...) -> ... -> operationN(input1, input2, ...)"'
            '\nPlease, don\'t use spaces between operations and inputs. Do not add any other information to the output.')
        self.operators = []
        self._add_operators(operators)
        self.instruction = instruction if (
                instruction is not None) else (
            '')

    def _add_operators(self, operators):
        self.operators += [OPERATIONS[operator] for operator in operators]

    def _generate_template(self):
        template = self.experiment_description + '\n'
        template += self.output_format + '\n'
        template += 'Available data operations are the following:\n'
        for operator in self.operators:
            template += f'\t{operator.__class__.__name__}: {operator.description}\n'
        template += self.instruction + '\n'
        return template

    def __str__(self):
        return self._generate_template()


@hydra.main(version_base=None,
            config_path='D:/PhD/LAAFE/src/optimization',
            config_name='cfg')
def main(cfg):
    llm_template = LLMTemplate(operators=cfg.llm.operators)
    print(llm_template)


if __name__ == '__main__':
    main()
