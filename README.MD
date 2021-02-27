[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <pre align="center"> 
    @@@  @@@  @@@@@@  @@@@@@@ @@@  @@@ @@@@@@@@ @@@@@@@  @@@ @@@  @@@ @@@@@@@@ 
    @@!  !@@ @@!  @@@   @!!   @@!  @@@ @@!      @@!  @@@ @@! @@!@!@@@ @@!      
    @!@@!@!  @!@!@!@!   @!!   @!@!@!@! @!!!:!   @!@!!@!  !!@ @!@@!!@! @!!!:!   
    !!: :!!  !!:  !!!   !!:   !!:  !!! !!:      !!: :!!  !!: !!:  !!! !!:      
     :   :::  :   : :    :     :   : : : :: ::   :   : : :   ::    :  : :: ::  
  </pre>

  <h3 align="center">kat-framework</h3>

  <p align="center">
    General video game A.I. framework
    <br />
    <a href="https://github.com/d33are/kat-framework"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/d33are/kat-framework">View Demo</a>
    ·
    <a href="https://github.com/d33are/kat-framework/issues">Report Bug</a>
    ·
    <a href="https://github.com/d33are/kat-framework/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#implemented-algorithms">Implemented Algorithms</a></li>
        <li><a href="#katherine-gameplay">Katherine gameplay</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
        <ul>
          <li><a href="#testing-the-installation">Testing the installation</a></li>
          <li><a href="#working-example">Working example</a></li>
          <li><a href="#optional-dependencies">Optional dependencies</a></li>
        </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains a general framework (Katherine) for playing video games with deep learning agents. This is a "home project" like framework, so it is currently under development, and it might have some bugs ( ¯\\\_(ツ)_/¯ ). It was not aimed for professional use, I'm just making this framework for educational purposes.

The main goal of the framework is to be simple as possible, and able to use it without any, or with very basic coding skills. The hole framework can be configured by `yaml` files, so for creating a new game scenario, you only need to create a configuration file and run it.

### Key Features

* API based modular design
* A.I. framework abstraction
* Configuration management
* Synchronous/Asyncronous and distributed learning
* fast low-level implementation

### Implemented Algorithms

* Deep-Q
* Double Deep-Q 
* Dueling Deep-Q 
* Double Dueling Deep-Q

### Katherine gameplay

* VizDoom - "basic" map:

| Gameplay                                                                                       |                                            Score MA                                             |
| ---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| <img src="/resources/vizdoom-basic/vizdoom_basic_gameplay.gif" width="320" title="hover text"> | <img src="/resources/vizdoom-basic/avg_score_vizdoom_basic.JPG" width="450" title="hover text"> |

* VizDoom - "defend the line" map:

| Gameplay                                                                                         |                                          Score MA                                          |
| ------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------: |
| <img src="/resources/vizdoom-defend/vizdoom-defend-gameplay.gif" width="320" title="hover text"> | <img src="/resources/vizdoom-defend/agent_total_score.JPG" width="450" title="hover text"> |

* OpenAI gym - Box2D - LunarLander:

| Gameplay                                                                                     |                                         Score MA                                         |
| -------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------: |
| <img src="/resources/lunar-lander/lunar-lander-gameplay.gif" width="320" title="hover text"> | <img src="/resources/lunar-lander/agent_total_score.JPG" width="450" title="hover text"> |

_These gameplay footages have been made without any code changes, only configuration was changed._

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of "how to check your python version."
  ```sh
  python --version
  ```
3.8 >= `python version` >= 3.7 will be fine.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/d33are/kat-framework.git
   ```
2. Install it with pip
   ```sh
   cd kat-framework
   ```
   ```sh
   pip install .
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Testing the installation

You can test your installation with the provided unit test package. This test is using dummy wrapper objects, so you don't need to install any games or A.I. frameworks.
   ```sh
   cd kat_framework_test
   ```
   ```sh
   python kat_framework_test.py
   ```

### Working example

* Hierarchical configuration:
  
  The runtime configuration is based on a hierarchical yaml file structure:
  ```
  project_dir
  | |- /config_dir
  | | |- default.yaml
  | | |- /your_specific_config
  | | | |- your_specific_config.yaml
  ```
  This will merge `config_dir/default.yaml` with `config_dir/your_specific_config/your_specific_config.yaml`.

  For more information about the hierarchical config, please refer to the [Documentation][himl-documentation].
  
* Minimum working configuration (default.yaml):
  ```
  global:
    game_class: kat_framework.games.katherine.dummy.DummyGame
    agent_class: kat_framework.agents.rand.RandomChoiceAgent
    driver_class: kat_framework.drivers.episode.SyncEpisodeDriver
    metrics_tracer_class: kat_framework.monitor.metrics.DummyTracer
    model_serializer_class: kat_framework.serialization.model.KatModelSerializer
    model_storage_driver_class: kat_framework.serialization.storage.DummyStorageDriver
    network_class: kat_framework.networks.models.random.RandomActionNetwork
    memory_class: kat_framework.memory.uniform.UniformMemory
    work_directory: training
  ```
  For more information about config values, please refer to the [Documentation][kat-config-documentation].
* Python entry point:
  
  Create a new python file and call the framework as follows:
  ```
  from kat_framework import KatherineApplication

  # default implementation
  FACTORY_CLASS = "kat_framework.core.factory.KatFactory"
  # default implementation
  CONFIG_CLASS = "kat_framework.config.config_handler.YamlConfigHandler"
  # this will points to the config_dir directory in the project directory
  CONFIG_URI = "file://localhost/config_dir" 
  ENABLE_LOGO = True

  if __name__ == "__main__":
      """
      Application entry point.
      """
      KatherineApplication.run(FACTORY_CLASS, CONFIG_CLASS, CONFIG_URI, ENABLE_LOGO) 
  ```

_For more detailed examples (with cli interface), please refer to the [Examples github repo][examples-github-repo]_

### Optional dependencies

* For tensorflow backend, please install tensorflow: [Installing Tensorflow][install-tensorflow]

* For playing with zDoom, please install VizDoom: [Installing zDoom][install-vizdoom]
* For playing with OpenAI gym games, please install gym: [Installing gym][install-gym]

_Implementations are included in the framework, just installations are needed._



<!-- ROADMAP -->
## Roadmap

See the [open issues][issues-url] for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open-source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

d33are - `d33are[at]gmail[dot]com`

Project Link: [https://github.com/d33are/kat-framework][project-link]


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/d33are/kat-framework.svg?style=for-the-badge
[license-url]: https://github.com/d33are/kat-framework/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/dkaszas
[issues-url]: https://github.com/d33are/kat-framework/issues
[project-link]: https://github.com/d33are/kat-framework
[himl-documentation]: https://github.com/adobe/himl
[kat-config-documentation]: https://github.com/d33are/kat-framework/blob/master/kat_framework/config/config_props.py
[examples-github-repo]: https://github.com/d33are/kat-examples
[install-tensorflow]: https://www.tensorflow.org/install/
[install-vizdoom]: https://github.com/mwydmuch/ViZDoom
[install-gym]: https://gym.openai.com/docs/