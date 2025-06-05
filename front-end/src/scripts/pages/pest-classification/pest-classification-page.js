import Camera from "../../utils/camera.js";
import { convertBase64ToBlob } from '../../utils/index.js';
import PestClassificationPresenter from "./pest-classification-presenter.js";
import * as GrowmateAPI from "../../data/api.js";

export default class PestClassificationPage {
  #presenter;
  #takenDocumentation = null;
  #camera;
  #isCameraOpen = false;

  async render() {
    return `
      <div class="main-feature-content">
        <div class="jumbotron">
          <img src="images/pest.jpg" alt="">
          <h1 class="title">Identifikasi <span>Hama Tanaman</span></h1>
        </div>

        <div class="container">
          <div class="description">
            <p>Pengguna dapat mengidentifikasi jenis hama yang menyerang tanaman hanya dengan mengunggah gambar bagian tanaman yang terdampak. Sistem akan secara otomatis menganalisis citra tersebut menggunakan teknologi AI, lalu menampilkan jenis hama, ciri-ciri umum serangan, serta saran penanganan yang dapat segera dilakukan untuk mencegah kerusakan lebih lanjut.</p>
          </div>

          <div class="recomendation">
            <div class="top-content">
              <i class="bi bi-search"></i>
              <p>Rekomendasi</p>
            </div>
            <div class="bottom-content main-recommendation-content">
              <div class="item">
                <p class="title">🌱 Daftar Penyakit yang Bisa Dideteksi</p>
                <p>Tampilkan jenis-jenis penyakit tanaman umum yang bisa dikenali oleh sistem.</p>
              </div>
              <div class="item">
                <p class="title">🤖 Keunggulan Teknologi yang Digunakan</p>
                <p>Didukung oleh model AI yang telah dilatih dengan ribuan gambar penyakit tanaman untuk akurasi deteksi yang tinggi.</p>
              </div>
              <div class="item">
                <p class="title">📸 Tips Foto yang Efektif untuk Deteksi</p>
                <ul>
                  <li>Ambil foto dengan pencahayaan yang baik.</li>
                  <li>Pastikan daun atau bagian tanaman terlihat jelas.</li>
                  <li>Hindari foto yang terlalu jauh atau terlalu dekat.</li>
                  <li>Usahakan untuk tidak ada objek lain yang menghalangi bagian tanaman yang ingin dideteksi.</li>
                </ul>
              </div>
              <div class="item">
                <form action="" id="form-upload">
                  <div class="upload-button-container">
                    <button type="button" class="btn camera-button">Buka Kamera</button>
                    <label for="image-file" class="btn file-button">Unggah Gambar</label>
                    <input type="file" id="image-file" name="image-file" style="display: none">
                  </div>
                  <div class="camera-container" id="camera-container">
                    <video id="camera-video" class="camera-video" autoplay playsinline>
                      Video stream not available.
                    </video>

                    <canvas id="camera-canvas" class="new-form__camera__canvas"></canvas>

                    <div class="new-form__camera__tools">
                      <select class="form-select" id="camera-select"></select>
                      <button id="camera-take-button" class="btn camera-take-button" type="button">
                        Ambil Gambar
                      </button>
                    </div>
                  </div>
                  <div class="image-preview" id="image-preview"></div>
                  <div class="detection-button-container" id="detection-button-container">
                    <button type="submit" class="detection-button"> Deteksi Sekarang</button>
                  </div>
                </form>
              </div>
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
    this.#presenter = new PestClassificationPresenter({
      view: this,
      model: GrowmateAPI,
    });

    this.#takenDocumentation = null;
    this.#setupForm();
  }

  #setupForm() {
    const imageFile = document.querySelector('#image-file');
    const imagePreview = document.querySelector('#image-preview');
    const formUpload = document.querySelector('#form-upload');

    imageFile ? imageFile.addEventListener('change', () => {
      const file = imageFile.files[0];

      if (!file) return;

      this.#takenDocumentation = {
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        blob: file,
        url: URL.createObjectURL(file),
      };

      const picture = this.#takenDocumentation;

      const html = `
        <button type="button" data-deletepictureid="${picture.id}" class="new-form__documentations__outputs-item__delete-btn">
          <img src="${picture.url}" alt="preview">
        </button>
      `;

      imagePreview.innerHTML = '';
      imagePreview.innerHTML = html;

      document.querySelector('button[data-deletepictureid]')?.addEventListener('click', (event) => {
        const pictureId = event.currentTarget.dataset.deletepictureid;
        const deleted = this.#removePicture(pictureId);
        if (!deleted) {
          console.log(`Picture with id ${pictureId} was not found`);
        }
        this.#populateTakenPictures();

        const detectionButton = document.querySelector('.detection-button');
        detectionButton ? detectionButton.classList.remove('active') : null;
        
        const output = document.querySelector('#output');
        const suggestion = document.querySelector('#suggestion');

        output ? output.classList.remove('active') : null;
        suggestion ? suggestion.classList.remove('active') : null;
      });
      
      const detectionButton = document.querySelector('.detection-button');
      detectionButton ? detectionButton.classList.add('active') : null;

      // this.#setupDecationButton();
    }) : null

    const cameraContainer = document.querySelector('#camera-container');
    const cameraButton = document.querySelector('.camera-button');
    cameraButton ? cameraButton.addEventListener('click', async(event) => {
      
      cameraContainer.classList.toggle('active');
      this.#isCameraOpen = cameraContainer.classList.contains('active');
      
      if (this.#isCameraOpen) {
        event.currentTarget.textContent = 'Tutup Kamera';
        this.#setupCamera();
        this.#camera.launch();
        return;
      }

      event.currentTarget.textContent = 'Buka Kamera';
      this.#camera.stop();
    }) : null

    formUpload ? formUpload.addEventListener('submit', async (event) => {
      event.preventDefault();

      let photoBlob = null;

      if (this.#takenDocumentation && this.#takenDocumentation.blob) {
        photoBlob = this.#takenDocumentation.blob;
      } else if (imageFile && imageFile.files.length > 0) {
        photoBlob = imageFile.files[0];
      }

      if (!photoBlob) {
        alert('Silakan ambil gambar atau unggah gambar terlebih dahulu.');
        return;
      }

      const data = { file: photoBlob };

      await this.#presenter.postPestPredict(data);
    }) : null
  }

  async #addTakenPicture(image) {
    let blob = image;

    if (image instanceof String) {
      blob = convertBase64ToBlob(image, 'image/png');
    }

    const newDocumentation = {
      id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
      blob: blob,
    };
    this.#takenDocumentation = newDocumentation;
  }

  async #populateTakenPictures() {
    if (!this.#takenDocumentation) {
      document.getElementById('image-preview').innerHTML = '';
      return;
    }

    const picture = this.#takenDocumentation;
    const imageUrl = URL.createObjectURL(picture.blob);

    const html = `
      <button type="button" data-deletepictureid="${picture.id}" class="new-form__documentations__outputs-item__delete-btn">
        <img src="${imageUrl}" alt="preview">
      </button>
    `;

    document.getElementById('image-preview').innerHTML = html;

    document.querySelector('button[data-deletepictureid]')?.addEventListener('click', (event) => {
      const pictureId = event.currentTarget.dataset.deletepictureid;
      const deleted = this.#removePicture(pictureId);
      if (!deleted) {
        console.log(`Picture with id ${pictureId} was not found`);
      }
      this.#populateTakenPictures();

      const detectionButton = document.querySelector('.detection-button');
      detectionButton ? detectionButton.classList.remove('active') : null;

      const output = document.querySelector('#output');
      const suggestion = document.querySelector('#suggestion');

      output ? output.classList.remove('active') : null;
      suggestion ? suggestion.classList.remove('active') : null;
    });
  }

  #removePicture(id) {
    if (this.#takenDocumentation && this.#takenDocumentation.id === id) {
      const deleted = this.#takenDocumentation;
      this.#takenDocumentation = null;
      return deleted;
    }

    return null;
  }

  #setupCamera() {
    if (this.#camera) {
      return;
    }

    this.#camera = new Camera({
      video: document.querySelector('#camera-video'),
      cameraSelect: document.querySelector('#camera-select'),
      canvas: document.querySelector('#camera-canvas')
    })

    this.#camera.addCheeseButtonListener('#camera-take-button', async () => {
      const image = await this.#camera.takePicture();
      // alert(URL.createObjectURL(image));
      await this.#addTakenPicture(image);
      await this.#populateTakenPictures();

      const detectionButton = document.querySelector('.detection-button');
      detectionButton ? detectionButton.classList.add('active') : null;

      // this.#setupDecationButton();
    })
  }

  setupDecationButton(data) {
    const output = document.querySelector('#output');
    const suggestion = document.querySelector('#suggestion');

    if (output) output.classList.add('active');
    if (suggestion) suggestion.classList.add('active');

    const outputTitle = document.querySelector('.output-title');
    const outputDesc = document.querySelector('.output-desc');
    const suggestionTitle = document.querySelector('.suggestion-title');
    const suggestionDesc = document.querySelector('.suggestion-desc');

    if (outputTitle) outputTitle.textContent = '🔍 Hasil';
    if (outputDesc) outputDesc.innerHTML = `Prediksi:  ${data.prediction} <br> Probabilitas: ${data.confidence}`;
    if (suggestionTitle) suggestionTitle.textContent = '💡 Saran';
    if (suggestionDesc) suggestionDesc.textContent = 'Praesent nec consectetur neque, vel efficitur neque. Vestibulum non turpis a nisl ornare imperdiet vitae et nisi. Duis tincidunt lobortis tellus ac viverra. Sed suscipit varius imperdiet. Ut condimentum pretium odio, non blandit metus viverra id. Proin sollicitudin metus nec volutpat commodo.';
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
    if (outputDesc) outputDesc.innerHTML = response.detail;
    if (suggestionTitle) suggestionTitle.textContent = '💡 Saran';
    if (suggestionDesc) suggestionDesc.textContent = 'Silakan unggah gambar yang dengan jelas memperlihatkan hama seperti serangga pada tanaman untuk membantu proses identifikasi secara akurat.';
  }

  predictFailed(message) {
    console.log(message);
  }
};
