import CropRecommendationPresenter from "./crop-recommendation-presenter.js";
import * as GrowmateAPI from "../../data/api.js";

export default class CropRecommendationPage {
  #presenter = null;

  async render() {
    return `
      <div class="main-feature-content">
        <div class="jumbotron">
          <img src="images/crop-recommendation.jpeg" alt="">
          <h1 class="title">Rekomendasi <span>Tanaman</span></h1>
        </div>
        
        <div class="container">
          <div class="description">
            <p>Pengguna dapat memperoleh rekomendasi tanaman yang paling sesuai untuk ditanam dengan memasukkan informasi kondisi tanah dan cuaca di sekitarnya. Sistem berbasis AI akan menganalisis data tersebut dan menampilkan daftar tanaman yang direkomendasikan, lengkap dengan tingkat kecocokan, kebutuhan dasar, serta tips awal untuk memulai penanaman secara efektif.</p>
          </div>

          <div class="recomendation">
            <div class="top-content">
              <i class="bi bi-search"></i>
              <p>Rekomendasi</p>
            </div>

            <div class="bottom-content main-recommendation-content">
              <div class="item">
                <p class="title">🔍 Data yang Digunakan</p>
                <p>Sistem ini menggunakan tujuh parameter, yaitu Nitrogen (N), Fosfor (P), Kalium (K), suhu (Temperature), kelembapan (Humidity), pH tanah, dan curah hujan (Rainfall) untuk memberikan rekomendasi tanaman yang paling sesuai dengan kondisi lingkungan Anda.</p>
              </div>
              <div class="item">
                <p class="title">🤖 Keunggulan Teknologi yang Digunakan</p>
                <p>Didukung oleh model machine learning yang dilatih menggunakan dataset pertanian, sistem ini mampu memberikan rekomendasi tanaman dengan akurasi tinggi berdasarkan input numerik.</p>
              </div>
              <div class="item">
                <p class="title">📝 Cara Menggunakan Sistem Ini</p>
                <p>Isi form di bawah dengan data dari 7 parameter lingkungan berikut:</p>
                <ul class="parameter-list">
                  <li>Nitrogen (N)</li>
                  <li>Fosfor (P)</li>
                  <li>Kalium (K)</li>
                  <li>Suhu (Temperature)</li>
                  <li>Kelembapan (Humidity)</li>
                  <li>pH Tanah</li>
                  <li>Curah Hujan (Rainfall)</li>
                </ul>
                <p>Setelah semua data diisi, lalu klik tombol "Dapatkan Rekomendasi".</p>
              </div>
              <div class="item">
                <form id="form-upload" class="crop-recommendation-form">
                  <div class="form-el">
                    <label for="nitrogen">Nitrogen (N)</label>
                    <input type="number" name="N" id="nitrogen" class="form-control input-value" placeholder="Contoh: 90" required>
                  </div>
                  <div class="form-el">
                    <label for="fosfor">Fosfor (P)</label>
                    <input type="number" name="P" id="fosfor" class="form-control input-value" placeholder="Contoh: 40" required>
                  </div>
                  <div class="form-el">
                    <label for="kalium">Kalium (K)</label>
                    <input type="number" name="K" id="kalium" class="form-control input-value" placeholder="Contoh: 60" required>
                  </div>
                  <div class="form-el">
                    <label for="suhu">Suhu (Temperature)</label>
                    <input type="number" name="temperature" id="suhu" class="form-control input-value" step="any" placeholder="Contoh: 26.5" required>
                  </div>
                  <div class="form-el">
                    <label for="kelembapan">Kelembapan (Humidity)</label>
                    <input type="number" name="humidity" id="kelembapan" class="form-control input-value" step="any" placeholder="Contoh: 80" required>
                  </div>
                  <div class="form-el">
                    <label for="ph">pH Tanah</label>
                    <input type="number" name="ph" id="ph" class="form-control input-value" step="any" placeholder="Contoh: 6.5" required>
                  </div>
                  <div class="form-el">
                    <label for="rainfall">Curah Hujan (Rainfall)</label>
                    <input type="number" name="rainfall" id="rainfall" class="form-control input-value" step="any" placeholder="Contoh: 200" required>
                  </div>
                  <div class="detection-button-container detection-button-container-2" id="detection-button-container">
                    <button type="submit" class="detection-button">Dapatkan Rekomendasi</button>
                    <button type="button" id="reset-button" class="reset-button">Kosongkan</button>
                  </div>
                </form>
              </div>
              <div class="item" id="loading"></div>
              <div class="item" id="output">
                <p class="title output-title"></p>
                <p class="output-desc"></p>
              </div>
              <div class="item" id="suggestion">
                <p class="title suggestion-title"></p>
                <p class="suggestion-desc"></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  async afterRender() {
    this.#presenter = new CropRecommendationPresenter({
      view: this,
      model: GrowmateAPI
    });

    this.#setupForm();
    this.#reset();
  }

  #setupForm() {
    const formUpload = document.querySelector('#form-upload');

    formUpload ? formUpload.addEventListener('submit', async (event) => {
      event.preventDefault();

      const data = {
        N: formUpload.elements.namedItem('N').value,
        P: formUpload.elements.namedItem('P').value,
        K: formUpload.elements.namedItem('K').value,
        temperature: formUpload.elements.namedItem('temperature').value,
        humidity: formUpload.elements.namedItem('humidity').value,
        ph: formUpload.elements.namedItem('ph').value,
        rainfall: formUpload.elements.namedItem('rainfall').value
      }

      await this.#presenter.postCropRecommendation(data);
    }) : null
  }

  #reset() {
    const resetButton = document.querySelector('#reset-button');
    const inputValues = document.querySelectorAll('.input-value');

    resetButton ? resetButton.addEventListener('click', () => {
      inputValues.forEach( inputValue => {
        inputValue.value = '';
      })
    }) : null
  }

  setupDecationButton(response) {
    const output = document.querySelector('#output');
    const suggestion = document.querySelector('#suggestion');

    if (output) output.classList.add('active');
    if (suggestion) suggestion.classList.add('active');

    const outputTitle = document.querySelector('.output-title');
    const outputDesc = document.querySelector('.output-desc');
    const suggestionTitle = document.querySelector('.suggestion-title');
    const suggestionDesc = document.querySelector('.suggestion-desc');

    if (outputTitle) outputTitle.textContent = '🔍 Hasil';
    if (outputDesc) outputDesc.innerHTML = `Rekomendasi:  ${response.data.recom_prediction} <br> Probabilitas: ${response.data.confidence}`;
    if (suggestionTitle) suggestionTitle.textContent = '💡 Saran';
    if (suggestionDesc) suggestionDesc.textContent = response.data.suggestion;
  }

  setupDecationButtonFailed(response) {
    const output = document.querySelector('#output');
    const suggestion = document.querySelector('#suggestion');

    if (output) output.classList.add('active');
    if (suggestion) suggestion.classList.add('active');

    const outputTitle = document.querySelector('.output-title');
    const outputDesc = document.querySelector('.output-desc');
    const suggestionTitle = document.querySelector('.suggestion-title');
    const suggestionDesc = document.querySelector('.suggestion-desc');

    if (outputTitle) outputTitle.textContent = '🔍 Hasil';
    if (outputDesc) outputDesc.innerHTML = response.error;
    if (suggestionTitle) suggestionTitle.textContent = '💡 Saran';
    if (suggestionDesc) suggestionDesc.textContent = 'Pastikan semua nilai pada formulir diisi dengan angka yang valid dan bernilai positif. Sistem tidak dapat memproses data jika ada input yang kosong, negatif, atau tidak sesuai format numerik. Cek kembali kolom seperti Nitrogen, Fosfor, Kalium, Suhu, Kelembapan, pH Tanah, dan Curah Hujan.';
  }

  predictFailed(message) {
    console.log(message);
  }

  showLoading() {
    const loading = document.querySelector('#loading');
    loading ? loading.innerHTML = `
      <i class="fas fa-spinner loader"></i>
    ` : null;
  }
  
  hideLoading() {
    const loading = document.querySelector('#loading');
    loading ? loading.innerHTML = `` : null;
  }

};