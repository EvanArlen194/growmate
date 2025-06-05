export default class GrowmatePage {
  async render() {
    return `<div class="growmate-content">
      <div class="container">
        <div class="jumbotron">
          <div class="description">
            <h1>GrowMate</h1>
            <p>GrowMate adalah aplikasi pintar yang membantu petani dan pecinta tanaman dalam mengidentifikasi hama, penyakit, serta memberikan rekomendasi perawatan secara akurat dan cepat.</p>
          </div>
          <div class="image-container">
            <img src="images/person.png" alt="">
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
          </div>
        </div>

        <div class="main-content">
          <section>
            <h3>Alasan Hadirnya GrowMate</h3>
            <p>Pertanian modern menghadapi tantangan seperti perubahan iklim, serangan hama yang tidak terdeteksi dini, dan minimnya akses informasi perawatan tanaman. GrowMate hadir sebagai solusi inovatif untuk menjawab kebutuhan ini dengan pendekatan berbasis teknologi yang mudah diakses oleh siapa saja.</p>
          </section>
          <section>
            <h3>Fitur Utama</h3>
            <div class="item">
              <h5>💡 Rekomendasi Perawatan</h5>
              <p>Setelah identifikasi dilakukan, GrowMate secara otomatis memberikan saran tindakan seperti jenis pestisida alami, pola penyiraman, hingga pemupukan yang sesuai dengan kondisi tanaman kamu.</p>
            </div>
            <div class="item">
              <h5>🔍 Klasifikasi Hama</h5>
              <p>Gunakan kamera ponsel untuk mengenali jenis hama yang menyerang tanaman secara instan. Dengan bantuan teknologi klasifikasi gambar, GrowMate memberikan identifikasi yang akurat agar kamu bisa segera mengambil tindakan.</p>
            </div>
            <div class="item">
              <h5>🦠 Klasifikasi Penyakit</h5>
              <p>Deteksi penyakit tanaman berdasarkan gejala visual seperti bercak, warna daun, atau tekstur yang berubah. GrowMate membantu menganalisis dan mengklasifikasikan penyakit agar penanganan lebih cepat dan tepat.</p>
            </div>
          </section>
          <section>
            <h3>Teknologi di Balik GrowMate</h3>
            <p>GrowMate menggunakan teknologi Computer Vision dan Machine Learning untuk mengidentifikasi hama dan penyakit tanaman secara akurat dari gambar yang diunggah pengguna. Kami terus melatih model kecerdasan buatan kami agar lebih cerdas seiring waktu berdasarkan data riil dari pengguna di lapangan.</p>
          </section>
          <section>
            <h3>Tim Pengembang</h3>
            <ul>
              <li>Evan Arlen Handy</li>
              <li>Muhamad Agus Faisal</li>
              <li>Trisya Nurmayanti</li>
              <li>Fitri Nailatul Khobibah</li>
              <li>Fachri Ibnu Falah</li>
              <li>Akbar Maulana Febriansyah</li>
            </ul>
          </section>
        </div>
      </div>
    </div>`
  }

  async afterRender() {
    // Add any additional functionality or event listeners here
    
  }
}