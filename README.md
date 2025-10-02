

Reader Lite
===========

The is a Web-based application used to interactively read a set of four books:

  1. Emma by Jane Austen
  2. The Iliad and the Odyssey by Homer
  3. Moby Dick by Herman Melville
  4. Walden Pond by Henry David Thoreau

Through the use of this tool the active reader (you) will enhance their use and understanding of the corpus. The whole thing kinda, sorta works like a back-of-the-book index but on steroids.


Requirements
------------

- Python >= v3.12 
    + Check with command `python --version`
    + Outdated or `command not found`? Use your existing/favorite Python Version Manager or follow [this guide]().
- Git 
    + Check with command `git --version`
    + `Command not found`? Follow [this guide]().
- Ollama
    + Check with command `ollama --version`
    + `Command not found`? [Download it](https://ollama.com/), open the downloaded file and follow install steps. Then open the app. 
    + **NOTE:** The Ollama app needs to be running in order to use it via command line!
- 5GB of free space on your computer (for Ollama models)

Installation
------------
0. Make sure you have the base requirements (see above).
1. Clone the source code using Git and change directory into it
    ```
    git clone https://github.com/ericleasemorgan/reader-lite.git && cd reader-lite
    ```
2. Install Ollama LLM models needed by Reader Lite
    ```
    ollama pull llama2:latest
    ollama pull nomic-embed-text:latest
    ```
3. Install necessary Python modules (including Flask, which will run the application)
    ```
    pip install .
    ```
4. Run the application with Flask
    ```
    flask --app reader_lite run --debug
    ```
    The app should now be running at open http://127.0.0.1:5000 in your web browser, and you ought to see something very similar to the following:


    <img width="600" height="349" alt="screenshot" src="https://github.com/user-attachments/assets/66b5ab89-1718-4a09-b2e9-0b12574e0989" />


Congratuations, you have successfully installed and launched Reader Lite. Whew.

Next time, just run the following command to pick up where you left off:

    flask --app reader_lite run --debug

While I can write rubust Python applications, I am still rusty on the writing of Python installation tools. Any help with the above instructions would be greatly appreciated!

---
Eric Lease Morgan &lt;eric_morgan@infomotions.com&gt;  
September 29, 2025