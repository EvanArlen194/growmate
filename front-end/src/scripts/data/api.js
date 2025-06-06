import CONFIG from '../config.js';

const ENDPOINTS = {
  PEST_PREDICT: `${CONFIG.PEST_BASE_URL}/predict`,
  DISEASE_PREDICT: `${CONFIG.DISEASE_BASE_URL}/predict-disease`,
  PLANT_RECOMMENDATION: `${CONFIG.CROP_BASE_URL}/predict/recom`
};

export async function getData() {
  const fetchResponse = await fetch(ENDPOINTS.ENDPOINT);
  return await fetchResponse.json();
}

export async function pestPredict({ file }) {
  const formData = new FormData();
  formData.set('file', file);

  try {
    const fetchResponse = await fetch(ENDPOINTS.PEST_PREDICT, {
      method: 'POST',
      body: formData,
    });

    const isJson = fetchResponse.headers
      .get('content-type')
      ?.includes('application/json');

    let json = {};

    if (isJson) {
      json = await fetchResponse.json();
    } else {
      json = { detail: 'Respons dari server tidak dikenali.' };
    }
  
    return {
      ...json,
      ok: fetchResponse.ok,
      status: fetchResponse.status,
    }
  } catch (error) {
    return {
      ok: false,
      status: 500,
      detail: 'Terjadi kesalahan jaringan atau server tidak merespon.',
    }
  }
}

export async function diseasePredict({ file }) {
  const formData = new FormData();
  formData.set('file', file);

  try {
    const fetchResponse = await fetch(ENDPOINTS.DISEASE_PREDICT, {
      method: 'POST',
      body: formData,
    });

    const isJson = fetchResponse.headers
      .get('content-type')
      ?.includes('application/json');

      let json = {};

      if (isJson) {
        json = await fetchResponse.json();
      } else {
        json = { detail: 'Respons dari server tidak dikenali.' };
      }

      return {
        ...json,
        ok: fetchResponse.ok,
        status: fetchResponse.status,
      }
  } catch (error) {
    return {
      ok: false,
      status: 500,
      detail: 'Terjadi kesalahan jaringan atau server tidak merespon.',
    }
  }
}

export async function plantRecommendation({ 
  N,
  P,
  K,
  temperature,
  humidity,
  ph,
  rainfall
}) {
  const formData = new FormData();
  formData.set('N', N);
  formData.set('P', P);
  formData.set('K', K);
  formData.set('temperature', temperature);
  formData.set('humidity', humidity);
  formData.set('ph', ph);
  formData.set('rainfall', rainfall);

  try {
    const fetchResponse = await fetch(ENDPOINTS.PLANT_RECOMMENDATION, {
      method: 'POST',
      body: formData
    });

    const isJson = fetchResponse.headers
      .get('content-type')
      ?.includes('application/json');

    let json = {};

    if (isJson) {
      json = await fetchResponse.json();
    } else {
      json = { detail: 'Respons dari server tidak dikenali.' };
    }

    return {
      ...json,
      ok: fetchResponse.ok,
      status: fetchResponse.status,
    }
  } catch (error) {
    return {
      ok: false,
      status: 500,
      detail: 'Terjadi kesalahan jaringan atau server tidak merespon.',
    }
  }
}