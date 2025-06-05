export default class DashboardPage {
  async render() {
    return `<div class="dashboard-content">
      <div id="carouselExample" class="carousel carousel-dark slide">
        <div class="carousel-indicators">
          <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
          <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="1" aria-label="Slide 2"></button>
        </div>
        <div class="carousel-inner">
          <div class="carousel-item active" data-bs-interval="2000">
            <img src="images/carousel-1.jpg" class="d-block w-100" style="" alt="...">
          </div>
          <div class="carousel-item">
            <img src="images/carousel-2.jpg" class="d-block w-100" style="" alt="...">
          </div>
        </div>
        <button class="carousel-control-prev" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
          <i class="bi bi-chevron-left" style="padding: 10px; background-color: rgba(255, 255, 255, .3) !important; font-size: 25px; color: #000;"></i>
        </button>
        <button class="carousel-control-next" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
          <i class="bi bi-chevron-right" style="padding: 10px; background-color: rgba(255, 255, 255, .3) !important; font-size: 25px; color: #000;"></i>
        </button>
      </div>

      <div class="container">
        <div class="description">
          <p>GrowMate adalah aplikasi berbasis AI yang membantu petani meningkatkan produktivitas dengan memberikan rekomendasi tanaman, deteksi hama, dan identifikasi penyakit secara cerdas.</p>
        </div>

        <div class="main-feature">
          <h2>Fitur Utama</h2>
          <div class="card-container">
            <div class="card">
              <div class="icon-container">
                <div class="background"></div>
                <i class="bi bi-crop"></i>
              </div>
              <div class="description">
                <h4>Rekomendasi</h4>
                <p>Dapatkan rekomendasi tanaman terbaik berdasarkan kondisi tanah dan cuaca.</p>
                <a href="#/crop-recommendation" class="btn more-button">Selengkapnya</a>
              </div>
            </div>
            <div class="card">
              <div class="icon-container">
                <div class="background"></div>
                <i class="bi bi-bug"></i>
              </div>
              <div class="description">
                <h4>Hama</h4>
                <p>Identifikasi jenis hama yang menyerang tanaman melalui gambar secara otomatis.</p>
                <a href="#/pest-classification" class="btn more-button">Selengkapnya</a>
              </div>
            </div>
            <div class="card">
              <div class="icon-container">
                <div class="background"></div>
                <i class="bi bi-leaf"></i>
              </div>
              <div class="description">
                <h4>Penyakit</h4>
                <p>Kenali jenis penyakit tanaman sejak dini untuk penanganan cepat dan tepat.</p>
                <a href="#/disease-classification" class="btn more-button">Selengkapnya</a>
              </div>
            </div>
          </div>
        </div>

        <div class="benefit">
          <h2>Manfaat Menggunakan GrowMate</h2>
          <div class="point">
            <p class="title">🧠 Rekomendasi Cerdas Berbasis AI</p>
            <p>GrowMate memberikan rekomendasi tanaman yang paling cocok berdasarkan kondisi tanah dan cuaca secara otomatis, tanpa perlu analisa manual.</p>
          </div>
          <div class="point">
            <p class="title">🐞 Identifikasi Hama Otomatis</p>
            <p>Pengguna cukup mengunggah foto, lalu sistem akan mendeteksi jenis hama yang menyerang dan memberikan saran penanganan.</p>
          </div>
          <div class="point">
            <p class="title">🌿 Deteksi Penyakit Dini</p>
            <p>Dengan deteksi dini berbasis gambar, petani dapat mengetahui penyakit tanaman lebih awal dan mencegah kerusakan lebih luas.</p>
          </div>
          <div class="point">
            <p class="title">📈 Meningkatkan Produktivitas Lahan</p>
            <p>Dengan informasi yang tepat dan cepat, petani bisa membuat keputusan lebih akurat dan hasil panen meningkat.</p>
          </div>
          <div class="point">
            <p class="title">⏱️ Hemat Waktu dan Tenaga</p>
            <p>Tidak perlu lagi cek satu per satu secara manual semua informasi bisa didapatkan langsung dari sistem dalam hitungan detik.</p>
          </div>
        </div>
      </div>

      <div class="tutorial">
        <div class="container">
          <h2>Cara Menggunakan Aplikasi</h2>
          <div class="gif-container" id="gif-container">
            <img src="images/gif-1.png">
            <div class="gif-caption">
              <p>GIF</p>
            </div>
          </div>
        </div>
      </div>

      <div class="big-gif-container" id="big-gif-container">
        <div class="container">
          <div class="modal-content">
            <div class="close-content">
              <div class="close-button" id="close-button"><i class="bi bi-x"></i></div>
            </div>
            <div class="gif-content">
              <div class="main-content">
                <img src="images/gif-1.png">
                <div class="gif-caption">
                  <p>GIF</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="container">
        <div class="contact">
          <h2>Hubungi Kami</h2>
          <div class="point">
            <p>GrowMate Team</p>
            <p>📬 Email: growmate.help@gmail.com</p>
            <p>📲 WhatsApp: +62 812-3456-0000</p>
          </div>
          <p>Butuh bantuan? Klik tombol di atas atau kirim email kapan saja. Kami siap membantu Anda!</p>
        </div>
      </div>

    </div>`
  }

  async afterRender() {
    this.enlargeGIF();
    this.closeBigGIF();
  }

  enlargeGIF () {
    const gifContainer = document.querySelector('#gif-container');
    gifContainer ? gifContainer.addEventListener('click', (event) => {
      event.stopPropagation();
      document.querySelector('#big-gif-container').classList.toggle('active');
    }) : null;
  }
  
  closeBigGIF() {
    const closeButton = document.querySelector('#close-button');
    
    closeButton ? closeButton.addEventListener('click', (event) => {
      event.stopPropagation();
      document.querySelector('#big-gif-container').classList.remove('active');
    }) : null;
  }
}