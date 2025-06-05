export default class DiseaseClassificationPresenter {
  #view;
  #model;

  constructor({ view, model }) {
    this.#view = view;
    this.#model = model;
  }

  async postDiseasePredict({ file }) {
    try {
      const data = { file: file }

      const response = await this.#model.diseasePredict(data);

      if (!response.ok) {
        this.#view.setupDecationButtonFailed(response);
        return;
      }

      // console.log('postDiseasePredict: response:', response);
      this.#view.setupDecationButton(response);
    } catch (error) {
      // console.log('postDiseasePredict: error:', error);
      this.#view.predictFailed(error.message);
    }
  }

}
