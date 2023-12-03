import hydra


class LLMTemplate:
    def __init__(self, operators):
        self.experiment_description = ''
        self.output_format = ''
        self.operators = operators
        self.interactions_example = ''

    def _add_operators(self, operators):
        self.operators += operators

    def _generate_template(self):
        template = self.experiment_description + '\n'
        template += self.output_format + '\n'
        template += 'operators:\n'
        for operator in self.operators:
            template += f'\t{operator}\n'
        template += self.interactions_example + '\n'
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
