import CropRecommendationPage from "../pages/crop-recommendation/crop-recommendation-page.js";
import DashboardPage from "../pages/dashboard/dashboard-page.js";
import DiseaseClassificationPage from "../pages/disease-classification/disease-classification-page.js";
import GrowmatePage from "../pages/growmate/growmate-page.js";
import PestClassificationPage from "../pages/pest-classification/pest-classification-page.js";

const routes = {
  '/': () => new DashboardPage(),
  '/growmate': () => new GrowmatePage(),
  '/crop-recommendation': () => new CropRecommendationPage(),
  '/pest-classification': () => new PestClassificationPage(),
  '/disease-classification': () => new DiseaseClassificationPage()
}

export default routes;