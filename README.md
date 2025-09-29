

Reader Lite
===========

The is a Flask-based Web application used to interactively read a set of four books:

  1. Emma by Jane Austen
  2. The Iliad and the Odyssey by Homer
  3. Moby Dick by Herman Melville
  4. Walden Pond by Henry David Thoreau

Through the use of this tool the active reader (you) will enhance their use and understanding of the corpus. The whole thing kinda, sorta works like a back-of-the-book index but on steroids.

Installation
------------

First, Reader Lite is a Python application. Open your terminal and run the following command, and version number greater than or equal to 3.12 ought to work:

  <code>python --version</code>

Second, Reader Lite requires the installation of Ollama, a tool making it easy to run generative-AI applications on your computer. Visit https://ollama.com/download and install Ollama. It is not hard. I promise.

Third, Reader Lite is configured to use two specific large language mnodels. Open your terminal and run the following two commands:

   1. <code>ollama pull llama2:latest</code>
   2. <code>ollama pull nomic-embed-text:latest</code>

Fourth, as if this writing, Reader Lite can only be downloaded from GitHub. Open your terminal and run the following command which will download the Reader Lite software:



---
Eric Lease Morgan &lt;eric_morgan@infomotions.com&gt;  
September 29, 2025

