import hydra

from src.optimization.data_operations.operation_aliases import OPERATIONS


class LLMTemplate:
    def __init__(
        self,
        operators,
        experiment_description=None,
        output_format=None,
        instruction=None,
        previous_evaluations=None,
    ):
        self.experiment_description = (
            experiment_description
            if (experiment_description is not None)
            else (
                "You should perform a feature engineering for the provided dataset to improve the performance of the model. "
            )
        )
        self.output_format = (
            output_format
            if (output_format is not None)
            else (
                "The output in an operation pipeline in following format:"
                "\nSTART FORMAT DESCRIPTION"
                '\n"operation1(df_column) , operation2(df_column_1, df_column_2) , operationN()"'
                "\nwhere empty brackets mean that operation is applied to all columns of the dataset."
                "\nPlease, don't use spaces between operations and inputs. Name operations exactly as they are listed in initial message. Do not add any other information to the output."
                "\nEND FORMAT DESCRIPTION"
            )
        )
        self.operators = []
        self._add_operators(operators)
        self.instruction = (
            instruction
            if (instruction is not None)
            else (
                "Based on the information provided, please, choose the operations you want to use in your pipeline and write them in the output format. Operation inputs have to match the columns of the dataset. Avoid repeating operation pipelines."
            )
        )
        self.previous_evaluations = (
            previous_evaluations
            if (previous_evaluations is not None)
            else ("Previous pipeline evaluations and corresponding metrics:\n")
        )

    def _add_operators(self, operators):
        self.operators += [OPERATIONS[operator] for operator in operators]

    def initial_template(self):
        template = self.experiment_description + "\n"
        template += self.output_format + "\n"
        template += "Available data operations are the following:\n"
        for operator in self.operators:
            template += f"\t{operator.__name__}: {operator.description()}\n"
        template += self.instruction + "\n"
        return template

    def instruction_template(self):
        template = self.instruction + "\n"
        return template

    def previous_evaluations_template(self):
        template = self.previous_evaluations
        return template

    def __str__(self):
        return self.initial_template()


@hydra.main(
    version_base=None, config_path="D:/PhD/LAAFE/src/optimization", config_name="cfg"
)
def main(cfg):
    llm_template = LLMTemplate(operators=cfg.llm.operators)
    print(llm_template)


if __name__ == "__main__":
    main()
