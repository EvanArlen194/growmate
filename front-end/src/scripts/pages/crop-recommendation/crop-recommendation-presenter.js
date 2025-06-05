export default class CropRecommendationPresenter {
  #view;
  #model;

  constructor({ view, model }) {
    this.#view = view;
    this.#model = model;
  }

  async postCropRecommendation({ 
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
  }) {
    try {
      const data = {
        N: N,
        P: P,
        K: K,
        temperature: temperature,
        humidity: humidity,
        ph: ph,
        rainfall: rainfall
      }

      const response = await this.#model.plantRecommendation(data);

      if (!response.ok) {
        this.#view.setupDecationButtonFailed(response);
        return
      }

      console.log('postCropRecommendation: response:', response);
      this.#view.setupDecationButton(response);
    } catch (error) {
      this.#view.predictFailed(error.message);
    }
  }
}