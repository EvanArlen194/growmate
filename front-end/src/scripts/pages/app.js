import routes from "../routes/routes.js";
import { getActiveRoute } from "../routes/url-parser.js";
import { transitionHelper } from "../utils/index.js";

class App {
  #content = null;
  #hamburgerMenu = null;
  #mobileNavigation = null;
  #navigation = null;
  #sidebarMenu = null;
  #mainMenu = null;

  constructor({ content, hamburgerMenu, mobileNavigation, navigation, sidebarMenu, mainMenu }) {
    this.#content = content;
    this.#hamburgerMenu = hamburgerMenu;
    this.#mobileNavigation = mobileNavigation;
    this.#navigation = navigation;
    this.#sidebarMenu = sidebarMenu;
    this.#mainMenu = mainMenu;

    this.#init();
  }

  #init() {
    this._setupHamburgerMenu();
    this._setupSidebarMenu();
    this._setupMainMenu();
  }

  _setupHamburgerMenu() {
    this.#hamburgerMenu.addEventListener('click', (event) => {
      event.stopPropagation();
      this.#mobileNavigation.classList.toggle('active');
      this.#navigation.classList.toggle('active');
    })
  }

  _setupSidebarMenu() {
    this.#sidebarMenu.forEach(menu => {
      menu.addEventListener('click', () => {
        this.#mobileNavigation.classList.remove('active');
      })
    });
  }

  _setupMainMenu() {
    this.#mainMenu.addEventListener('click', () => {
      this.#mobileNavigation.classList.remove('active');
    })
  }

  async renderPage() {
    const url = getActiveRoute();
    const route = routes[url];

    const page = route();

    const transition = transitionHelper({
      updateDOM: async () => {
        this.#content.innerHTML = await page.render();
        page.afterRender();
      },
    })
    

    transition.ready.catch(console.error);
    transition.updateCallbackDone.then(() => {
      scrollTo({ top: 0, behavior: 'instant' });
    });
  }
}

export default App;