export default class PestClassificationPresenter {
  #view;
  #model;

  constructor({ view, model }) {
    this.#view = view;
    this.#model = model;
  }

  async postPestPredict({ file }) {
    try {
      const data = { file: file }

      const response = await this.#model.pestPredict(data);

      if (!response.ok) {
        this.#view.setupDecationButtonFailed(response);
        return;
      }

      this.#view.setupDecationButton(response.data);
    } catch (error) {
      // console.log('postPestPredict: error:', error);
      this.#view.predictFailed(error.message);
    }
  }
}
