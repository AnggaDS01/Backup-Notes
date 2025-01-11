# Label Studio Installation & Auto Annotation: The Ultimate Guide

![image](https://github.com/user-attachments/assets/4ca4f684-fded-4060-a78c-a1a568337ce8)

---

## Manage Labeling Data More Efficiently
In the world of machine learning, data is the primary fuel. However, we all know that the data labeling process can be time-consuming and sometimes frustrating. This is where Label Studio comes in as a solution, offering flexibility to manage data more easily. You might ask, why Label Studio? Okay, here's my personal opinion:

### Advantages of Label Studio
1. **Supports Multiple Data Types** \
You can label data in various formats, including images, text, audio, and video. This makes Label Studio flexible for different AI projects, whether it's image classification, object detection, or NLP tasks.

2. **Auto Annotation Feature** \
If you have a pre-trained AI model, this feature can help automatically label data. It's perfect for saving time when dealing with large datasets.

3. **Open Source & Free**\
Since it's open source, you can use it for free, and the community is quite active. If you encounter issues or want new features, you can check [**GitHub**](https://github.com/HumanSignal/label-studio) or contribute yourself.

4. **Easily Customizable**\
Label Studio gives you full control to design the labeling interface according to your needs. For example, horizontal label views, adding extra tools, or integrating your custom AI model.

5. **Strong AI Ecosystem Support**\
You can integrate Label Studio with other popular tools like TensorFlow, PyTorch, and Hugging Face. The labeled data can even be directly processed in your AI pipeline.

To be fair, I'll also share my personal opinion regarding the drawbacks of Label Studio:

### Disadvantages of Label Studio
1. **Requires a Non-Simple Setup**\
To use advanced features like **Auto Annotation**, you need to set up your own **model inference server**. This can be challenging if you're not familiar with **server or GPU environments**.

2. **Performance on Large Datasets**\
Label Studio sometimes struggles **with extremely large datasets**, especially if you're running it on a system **without a GPU or high specifications**.

3. **Basic UI/UX**\
While functional, its **user interface (UI)** can feel stiff and less **user-friendly** compared to some paid tools.

4. **Limited Documentation for Specific Cases**\
The **official documentation** is helpful for basic setup, but if you're looking to configure more complex setups (like **integrating custom models** or **inference servers**), you'll need to rely on **trial-and-error** and additional research.

5. **Not Suitable for Non-Technical Users**\
If the target audience includes **non-technical users**, Label Studio might feel complicated without guidance from someone with **technical expertise**.

Therefore, in this article, I'll guide you **step-by-step** to install Label Studio and provide instructions on how to enable the **Auto Annotation** feature. With this feature, you can save **time** and improve **efficiency** in your data labeling projects.

To enhance **inference/prediction performance** for Auto Annotation, I decided to utilize a GPU. Why? Because GPUs significantly speed up **AI model computations** compared to regular CPUs. This setup helps save time and energy, especially for **large-scale projects**.

In this article, I'll explain how to install Label Studio on **WSL2**, including setting up the **NVIDIA drive**r for GPU access. Additionally, I'll use **Docker** to run the inference server. Docker allows you to create a consistent and easily deployable **isolated environment**, while Miniconda will be used for managing dependencies.

Why **WSL2**? Because I want to take advantage of **Linux's flexibility and performance** for AI tasks without giving up the convenience of **Windows**. I chose Miniconda to keep my **Python environment lightweight** and isolated from **dependency conflicts**.

By the end of this article, I hope you not only understand how to set it up but also the reasoning behind each step. Let's start our journey toward smarter and more efficient data labeling! Moving forward, we'll:
1. Explain **step-by-step** how to set up the **NVIDIA Driver**.
2. Explain **step-by-step** how to set up **WSL2**.
3. Explain **step-by-step** how to install **Docker**.
4. Explain **step-by-step** how to install **Miniconda** on WSL2.
5. Explain **step-by-step** how to install **Label Studio**.
6. Explain **step-by-step** how to set up **Local Storage** in Label Studio.
7. Explain **step-by-step** how to set up **Auto Annotation** in Label Studio.

If you're wondering why I didn't mention **MacOS**, it's because as far as I know, **MacOS doesn't support CUDA**, so it can't be used for acceleration in inference/predictions (cmiiw 😅).

Before we move on to the installation steps, it's important to note some prerequisites.

### System requirements:
* **Windows WSL2** - Windows 10 19044 or higher (64-bit). To check, press the **Windows button (⊞ Win)** and type **"About your PC".**

![image](https://github.com/user-attachments/assets/1a097952-11b8-42dd-8f14-217c2f5d7a2b)

* **RAM Capacity** - Ideally, your PC/Laptop should have **8GB RAM**, but **16GB RAM** or more is recommended.

### Software requirements
* **Python version ≥ 3.9** or ≤ **3.12** (we'll install it using **Miniconda**).
* **NVIDIA® GPU drivers**, version **>= 528.33** for WSL (👉 [**check the info here**](https://www.nvidia.com/en-us/drivers/), and you might be required to log in).
* Docker Desktop for Windows (👉 [**check the info here**.](https://docs.docker.com/desktop/setup/install/windows-install/)).

Why did I specify the version of the **NVIDIA driver**? Because if you need to train models using **TensorFlow** or **PyTorch**, the driver is still compatible. This step prevents errors caused by version incompatibility. If you want to know how to install **TensorFlow** and **PyTorch** with GPU support, you can check the 👉 [**tutorial here**](https://medium.com/@anggadwisunarto3/how-to-install-tensorflow-and-pytorch-with-gpu-support-a-beginner-friendly-guide-1dae716652cd).

### Hardware requirements
* **NVIDIA® GPU card** (remember, it has to be **NVIDIA**. As far as I know, at the time of writing this article, **AMD GPUs don't support Deep Learning tasks, cmiiw**).

---

## Setup NVIDA Driver

### Why Do You Need to Set Up NVIDIA Driver?

Think of the **NVIDIA Driver** as the "communication brain" between your operating system and the **NVIDIA GPU** installed on your system. Without installing this driver, your GPU would just be a decorative piece in your PC/laptop. So, the driver is absolutely essential to activate the GPU features, especially **CUDA**, which we'll use for accelerating models created with **TensorFlow** or **PyTorch**.

### Steps to Set Up NVIDIA Driver
Alright, now let's dive into the detailed steps. Don't worry, bro, it's easy as long as you follow the steps one by one.
First, you need to download the NVIDIA driver that matches your GPU. Go directly to NVIDIA's official website:
👉 [**NVIDIA Driver**](https://www.nvidia.com/en-us/drivers/).

On the site, you'll see a form to select your driver. Fill it out based on your **GPU specifications**. If you're unsure about your GPU's specs, press the **Windows key (⊞ Win)** on your keyboard, type **"Device Manager"**, and you'll see the type of GPU you're using.

![image](https://github.com/user-attachments/assets/8686a3f4-0d49-4e5a-95ff-03fe2e15af5e)

Once you've identified your GPU, go back to the NVIDIA website to download the driver. Fill out the form with the details matching your GPU.

![image](https://github.com/user-attachments/assets/332e4a98-7957-446a-93e4-c203b446ba0c)

For example, here's what it looks like for my laptop's GPU. Not too powerful, I know 😅… Then, click Find, and it will take you to the next page.

![image](https://github.com/user-attachments/assets/785cc6e6-756f-4146-8540-cf9ce0fd377f)

![image](https://github.com/user-attachments/assets/48d9a94b-c0bd-47c9-be56-703c7d66cddd)

Simply click **Download**, wait for the download to complete, and then click the installer file. If you don't want to mess with configurations, just keep clicking **OK** and **Next**, and leave the settings as default. Sorry, I don't have screenshots for this because I've already installed it. Instead, you can check out this tutorial (hope it helps).

[<img src="https://img.youtube.com/vi/Yd8IYK9NwJA/hqdefault.jpg"/>](https://www.youtube.com/embed/Yd8IYK9NwJA?t=220s)

Focus on the installation duration. Note that the video is from 2022, so the UI on NVIDIA's website might look different in 2024. Once the NVIDIA driver is installed, confirm its installation by typing the following command in Powershell or CMD:

```bash
nvidia-smi
```

If you see an output like this, it means your NVIDIA driver is successfully installed:

![image](https://github.com/user-attachments/assets/dd65588e-28fb-4153-b2ca-b5af550cc92f)

Now that your NVIDIA Driver is set up, let's move on to setting up WSL2 🐧!

---

## Setup WSL2
Okay, after the NVIDIA driver is installed, let's move on to the next step: setting up **WSL2** (Windows Subsystem for Linux 2). **WSL2** is super important because we will be running **Label Studio** in a Linux environment, but still from Windows. In short, you get the Linux experience without needing a dual-boot.

Why WSL2?
You might be wondering, why use WSL2? Here's the deal, bro, WSL2 is more advanced than the previous version (WSL1). The differences are:
* **Performance: WSL2** uses the actual Linux kernel, so it's faster.
* **GPU Support**: This is the most important part! **WSL2** supports GPU acceleration via **CUDA**.
* **Windows-Linux Integration**: You can access Windows files from Linux and vice versa.

Before setting up WSL2, there are a few things you need to do:
1. **Enable Developer Mode**: Just search for it in the Windows menu by pressing the Windows button **(⊞ Win)**, then type **"Developer Settings"**, and switch it to **"On"**.

![image](https://github.com/user-attachments/assets/5229021f-8d59-4935-a46b-1d1044adfa96)

For newer versions of Windows (Windows 11 and some newer Windows 10), Developer Mode isn't that crucial because virtualization features are usually enabled by default. But if you are using an older version or run into errors during the **WSL2** setup, this step will be a lifesaver. So, better safe than sorry!

2. Then press the Windows button (⊞ Win), search for "Turn Windows features on or off", and select the menu that appears.

![image](https://github.com/user-attachments/assets/c959b20e-d171-44c3-806a-a988ec501427)

Check the boxes for **"Virtual Machine Platform"** and **"Windows Subsystem for Linux"**, then click **OK**. After that, you'll be asked to restart your PC/laptop. Just follow the instructions and restart your device.

Once done, open **PowerShell** or **CMD** in Windows and type the following:

```bash
wsl
```

If the output is like this:

```
Windows Subsystem for Linux has no installed distributions.
```

```bash
Use 'wsl.exe --list --online' to list available distributions
and 'wsl.exe --install <Distro>' to install.
Distributions can also be installed by visiting the Microsoft Store:
https://aka.ms/wslstore
Error code: Wsl/Service/CreateInstance/GetDefaultDistro/WSL_E_DEFAULT_DISTRO_NOT_FOUND
```

This means the Linux distro (OS distribution) is not installed yet. Now, check the **WSL** version by typing:

```bash
wsl --version
```

The output will tell you the version:

```
WSL version: 2.2.4.0
Kernel version: 5.15.153.1-2
WSLg version: 1.0.61
MSRDC version: 1.2.5326
Direct3D version: 1.611.1-81528511
DXCore version: 10.0.26091.1-240325-1447.ge-release
Windows version: 10.0.19045.5131
```

Now, if the WSL version is not `2.x.x.x`, that means your WSL is not on version **2** yet. Upgrading to version **2** is super easy, just type:

```
wsl --set-default-version 2
```

But if you've already typed the command above and the version is still at `1.x.x.x`, try looking it up on YouTube or researching **"why WSL isn't upgrading to version 2"**. Once you figure that out, we can move on to the next step.

Now that **WSL2** is set up, we still need to install the Linux distro in Windows, because frameworks like **TensorFlow** or **PyTorch** only support Ubuntu, so we'll install **Ubuntu**.

Here's how: Press the Windows button **(⊞ Win)**, then type **"Microsoft Store"** in the search bar, and search for **"Ubuntu"**.

![image](https://github.com/user-attachments/assets/3e69194d-107d-4738-a264-c13151031881)

Choose the one with no version number in the title because that's the latest and most stable version, then just install it. Click **"Open"** once it's done downloading. You will be asked to create a username and password. This is to set up your Linux environment in Windows. So, the username and password you create will actually be your Linux system credentials. You'll need them every time you access **WSL2** through the terminal.

For example, if you need to install a package or do something that requires admin permission, you will be asked to enter this password. So, make sure you remember it well, bro, don't just type anything!

Now that you have Ubuntu on Windows, every time you want to access Ubuntu in Windows, you'll need to go to PowerShell/CMD and type wsl. If you want to check the version of Ubuntu, you can type:

```bash
lsb_release -a
```

output:

```
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.1 LTS
Release:        24.04
Codename:       noble
```

This is the version of **Ubuntu** I have, as of the time this article was written.

Okay, before we move on to the next step, we need to **update** and **upgrade** the packages in **Ubuntu**. Why? Because when you first install **WSL2**, the Ubuntu you get is really fresh, like a default setup. So, many packages inside it might not have been updated or might not even be installed at all. It's like buying a new phone, but the apps are all in their old versions - they need to be updated first to make sure the features are working well, right?

The same goes here. You need to update the package list so the system knows which updates are available, and then upgrade so all the applications are up to date. This is super important, bro, especially if you plan to install frameworks like TensorFlow or PyTorch later.

First, type this in your Ubuntu terminal:

```bash
cd ~
```

This means you're going to the root directory in **Ubuntu**. Actually, it's not mandatory, but when you enter Linux via the terminal and type `wsl`, you're actually not in a Linux environment but in Windows. Confused? 😅 Let me show you the picture.

![image](https://github.com/user-attachments/assets/41e84fb0-bcb1-4bdc-87b5-445a913d3b9f)

You can see for yourself, the path `/mnt/c/Users/user` is still in Windows, so if you want to download a file from the outside, for example from Google, it will eventually be in the user folder here (I named my laptop user, yours will be different). Once done, you can type:

```bash
sudo su
```

So, this helps you enter admin mode, so you have full access to make changes or installations in your Ubuntu system. It's like you're the "boss level" who can manage everything without any restrictions. This mode is super important if you want to install packages or edit system files that are usually restricted to administrators.

Now, to **update** and **upgrade** your packages, you can type:

```bash
apt update && apt upgrade -y
```

Then wait until it finishes, usually it doesn't take long, depending on your internet speed. Next, let's move on to the step to install Docker Desktop for Windows 🐋.

---

## Install Docker
Next, you need to install Docker. You can watch this video for the installation process. I recommend watching it until the end first, before practicing, or check out 👉 [**this documentation**](https://docs.docker.com/desktop/setup/install/windows-install/), it's very complete.

[<img src="https://img.youtube.com/vi/ZyBBv1JmnWQ/hqdefault.jpg"/>](https://www.youtube.com/embed/ZyBBv1JmnWQ)

> IMPORTANT❗: If some of the steps for installing Docker Desktop above have already been done in the previous steps of this article, don't do them again. Just do them once (for example: enabling virtual machine in Windows, WSL, and so on).

Once it's installed, open **Docker Desktop** from the Windows menu, then go to **`Settings >> Resources >> WSL Integration`**, and enable it according to the Linux distribution you chose. For example, if I'm using Ubuntu, then enable that.

![image](https://github.com/user-attachments/assets/3e56a1ac-b841-49f5-9a64-12f79d5435d5)

![image](https://github.com/user-attachments/assets/50204f93-24ce-43a1-9e40-0a9166887ce2)

Once you're done, let's move on to the next step: installing Miniconda 🐍.

---

## Install Miniconda in the WSL2 Distro
Before we get into the installation steps for Miniconda, there's something important to check. Since we've already installed Ubuntu and Docker, we need to make sure that the default WSL points to Ubuntu, not Docker. Why? Because if the default is set to Docker, when you type the wslcommand, you'll end up in Docker instead of Ubuntu.

It's easy to check. Type the following command in Powershell/CMD:

```bash
wsl --list --verbose
```

If the default WSL still points to Docker, the output will look like this:

```
NAME              STATE           VERSION  
* docker-desktop    Stopped         2  
  Ubuntu            Running         2
```

Look for the asterisk (*) next to **docker-desktop**? That means Docker is currently the default **WSL**. To change it so the default is **Ubuntu**, just type the following command:

```bash
wsl --set-default Ubuntu
```

Then, check again with the same command:

```bash
wsl --list --verbose
```

Now, the output should look like this:

```
NAME              STATE           VERSION  
* Ubuntu            Running         2  
  docker-desktop    Stopped         2
```

The asterisk (*) next to Ubuntu indicates that Ubuntu is now the default WSL. With this, every time you type `wsl`, you will directly enter Ubuntu, not Docker.

Now, let's move on to the next important part, which is installing **Miniconda**. You might be asking, "Why use Miniconda instead of Anaconda, or why not just install the packages directly using `pip`?"

The answer is that **Miniconda** is lighter than **Anaconda**, and both **Miniconda** and **Anaconda** are like "logistics managers" for managing all your Python library needs in Ubuntu. With **Miniconda**, you can create a virtual environment specifically for your project. So, if you want to install **TensorFlow**, **PyTorch**, or other libraries with **different versions** for your project needs, they won't overlap or cause conflicts. You can also easily install the Python version you need using the **conda prompt** when setting up a virtual environment. This keeps everything neat and safe, and we will use this method to create a separate environment when installing **Label Studio**.

A simple analogy: **Miniconda** is like having a separate toolbox for each project. So, if you're working on project "A", you only open the toolbox for that project. The tools won't get mixed up with the toolbox for project "B". Convenient, right?

Okay, before we move on to the installation process, make sure you've already done the **"update"** and **"upgrade"** for your Ubuntu as we discussed earlier. Once done, let's proceed with the command to download the **Miniconda installer**. Ensure you're in the **root path** of Linux by typing the following command:

```
cd ~
```

This ensures the file you download is stored in the root of Linux, not in Windows. Then type `sudo su` in the Ubuntu terminal, but if you've already used sudo su, you don't need to do it again. To download Miniconda, type the following command:

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
```

This command will download the latest version of the Miniconda installer directly from their official repository. So, you don't have to worry about security or outdated versions. After downloading Miniconda using the previous command, make sure the Miniconda installer file is located in the root of your Ubuntu. To check, type ls in the terminal. If the file isn't there, the next command won't work because it won't find the installer file.

Next, you'll run the installation script. It's really simple, just type this command in your Ubuntu terminal:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

When you execute this command, the installation process will begin, and you'll see several prompts that need to be answered. Here's the explanation for each prompt in case they appear:

```
Do you accept the license terms? [yes|no]
```

This one appears because **Miniconda** wants to confirm that you agree with its license terms. No worries, nothing strange here. Just type yes and press **Enter**. Next, if you see:

```
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
```

This is to set up **Miniconda** automatically on your shell. So, if you type **conda** in the terminal, it will be recognized. Again, just type yes here to have everything set up automatically without any hassle.

> IMPORTANT❗: After installation is complete, you must restart your terminal for all the changes to take effect. It's simple: just exit the terminal by typing `exit`, then open the terminal again and run `wsl`.

If everything goes well, you'll see (base) next to your username in the terminal. This means Miniconda is active, and you now have the base environment set as default. From here, you're ready to create specific environments based on your needs… 👌

![image](https://github.com/user-attachments/assets/10fef643-bfdf-4704-b6c7-c27bd4a60252)

If it doesn't show up, try going over the previous steps again, because we need Miniconda to set up environments with specific versions of Python. To confirm if Conda is installed, type this command:

```
conda --version
```

![image](https://github.com/user-attachments/assets/74c097be-c978-4627-94df-71d749821cbc)

My conda version is `24.9.2`, as of when this article was written. And that's it, we've installed **Miniconda**, now let's move on to installing **Label Studio** 🚀.

---

## Install Label Studio
Alright, now that the **Miniconda** setup is done, it's time to install **Label Studio** on our Ubuntu. But before that, we need to create a **virtual environment**. Why? To prevent any dependencies for **Label Studio** from conflicting with other projects you might be working on in the same system. This virtual environment is like a **"private space"** for specific applications, so everything stays neat and safe.

The steps are simple:

1. Create a Virtual Environment by typing the following command in your terminal:

```bash
conda create -n label_studio python=3.11 -y
```

This command will create a new environment named label_studio with Python version 3.11. Don't forget to activate this environment before moving forward:

```bash
conda activate label_studio
```

Once you run the above command, you should see something like the one below. You'll notice that the **Ubuntu username** changes from `base` to `label_studio`, which means your virtual environment is now active.

![image](https://github.com/user-attachments/assets/2795525a-1581-4bf7-bda7-592feba6ad1c)

Next, upgrade pip by typing the following command:

```
pip install --upgrade pip
```

2. **Install Label Studio**
After the environment is active, you can install Label Studio with:

```
pip install -U label-studio
```

Wait for the installation to finish. Once it's done, you can check if Label Studio was successfully installed with this command:

```
label-studio --version
```

3. **Jalankan Label Studio**
If the installation was successful, you can directly run **Label Studio** with:

```
label-studio
```

This will automatically open the **Label Studio** app in your browser. You can use **Chrome**, but I usually use **Microsoft Edge**. From here, you can start exploring and setting up your first project. The default port for **Label Studio** is 8080, so just type that in the address bar of your browser.

![image](https://github.com/user-attachments/assets/5311d984-63da-46dd-baab-1dec4462bf55)

If this is your first time installing, you'll be asked to sign up. Just follow the instructions, and once you're done, you can play around with a small dataset to get a feel for it. If you want to dive deeper into using it, you can check the guide 👉 [**here**](https://labelstud.io/guide/labeling).

Oh, and if you're still curious about the installation process, and whether it can be installed without pip, you can visit the site 👉 [**here**](https://labelstud.io/guide/install.html). Also, if you want to change the default port, check out the documentation 👉 [**here**](https://labelstud.io/guide/start#Run-Label-Studio-on-localhost-with-a-different-port). However, as I mentioned at the start of this article, the Label Studio documentation isn't very detailed, so you'll need to explore on your own, which means trial and error - just like I did when creating this tutorial.

Next, I'll discuss how to set up **local storage access** in **Label Studio**. Let's move on to the next step!

---

## Setup Local Storage di Label Studio
At this point, **Label Studio** is ready to use. But there's one important thing we need to set up, especially if you have a large and extensive dataset: **Local Storage**.

Why is Local Storage important? Because if you directly upload a huge dataset to **Label Studio**, it can result in errors. For example, you might see an error like this:

![image](https://github.com/user-attachments/assets/9947d6e4-4b7e-48de-bf99-94ef89f4c5d6)

As you can see, there's an error message in red, telling you that the uploaded file is too large. So, what's the solution? Do we need to upload it gradually? Sure, but that would be super inefficient. That's why I'm going to share a tip on how to handle errors like the one shown above.

Instead of uploading the dataset directly, we can take advantage of Local Storage in Label Studio. This feature allows Label Studio to access your dataset stored on your computer without uploading everything. How convenient is that?

You might wonder, why not use cloud storage instead? The answer is that cloud storage usually comes with a cost, especially if your dataset is large. You might face additional charges for storage and data access, which can get expensive, especially for beginners or small teams with limited budgets. However, if you're not constrained by a budget, I'd recommend using cloud storage. If you want to learn how to connect cloud storage to **Label Studio**, you can check the documentation 👉 [**here**](https://labelstud.io/guide/storage).

Now, back to the original case where the dataset we want to import is too large. Open your terminal (I recommend installing **Windows Terminal**). To do this, go to the **Microsoft Store** from the Windows menu (⊞ Win), then type "Windows Terminal" in the search bar and install it.

![image](https://github.com/user-attachments/assets/58bb9332-fd40-4c25-b9c1-a90cf43d20e5)

Why use **Windows Terminal**? Because you can open a new tab within the terminal without having to open a separate terminal window, which keeps everything neat and organized. Here's an example:

![image](https://github.com/user-attachments/assets/9c34bf67-117c-457e-8389-8d1b80d61ecc)

Next, open the **Ubuntu terminal** by typing wsl. Once the terminal is open, we need to create an **environment variable** in **Ubuntu**. Don't worry, you don't need to enter the `label_studio` virtual environment or run **Label Studio** just yet.

Before continuing, create a specific folder to store your dataset. You can name it however you like, depending on your needs. In my case, I created a folder like this:

```bash
mkdir -p Workspace/Label_Studio/Datasets
```

Then, set up the environment variable by entering the following command to access the `~/.bashrc` file:

```bash
nano ~/.bashrc
```

This file is the configuration file for the **Bash shell** (Bourne Again Shell) used in Linux or Unix-based systems, including your **WSL**. Its function is to store various settings and environment variables that will run automatically every time you open a **Bash** terminal or log into a **Bash** session interactively.

Once you're in the `nano` **editor**, scroll down to the bottom and type or copy the following script:

```
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/anggads01/Workspace/Label_Studio/Datasets
```

![image](https://github.com/user-attachments/assets/2ba5c38f-6adc-44be-9349-58c610b9e4c1)

Remember to adjust the path in `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`.

```
/home/anggads01/Workspace/Label_Studio/Datasets
```

It should match the folder where your dataset is saved. So the path above refers to the `Datasets` folder as the parent folder, which will contain subfolders of your dataset. Confused? Okay, now take a look at the image below:

![image](https://github.com/user-attachments/assets/91312221-e8ca-43b6-9b49-88458634e96c)

You can see that in the `Datasets` folder, there are subfolders called `Fruit_Images` and `Receipt_Images`, which are my datasets containing the images to be labeled.

To save the file in the nano editor, press `Ctrl+O`, then press `Enter`, and to exit the `nano` editor, press `Ctrl+X`. Quick tip if you want to know nano editor shortcuts.

```
Navigation Shortcuts
Ctrl + A: Go to the beginning of the line.
Ctrl + E: Go to the end of the line.
Ctrl + Y: Scroll up (Page Up).
Ctrl + V: Scroll down (Page Down).
Alt + /: Go to the end of the file.
Alt + : Go to the start of the file.
Search and Replace Shortcuts
Ctrl + W: Search for text.
Ctrl + : Search and replace text.
Alt + W: Find the next occurrence.
Text Editing Shortcuts
Ctrl + K: Cut the current line.
Ctrl + U: Paste the cut line.
Ctrl + J: Justify text (Align left and right).
Alt + 6: Copy the highlighted text.
Undo and Redo
Alt + U: Undo (Revert the last change).
Alt + E: Redo (Restore the undone change).
File Shortcuts
Ctrl + O: Save the file (Write Out).
Ctrl + X: Exit the editor.
Ctrl + R: Insert a file into the text you're editing.
Line Manipulation Shortcuts
Ctrl + T: Activate spell check.
Ctrl + ^: Start marking text.
Alt + ]: Indent the line.
Alt + [: Remove the indentation of the line.
Display Help
Ctrl + G: Show the help menu.
Pro Tips
1. Shortcut Combinations: For example, use Ctrl + K to cut several lines, then paste them with Ctrl + U.
2. Highlight Text Block: Press Ctrl + ^, then use arrow keys to select text. Use Ctrl + K to cut or Alt + 6 to copy.
3. Quick Search and Replace: Use Ctrl + , then input the text you want to search and replace.
```

Next, save the changes and run this command to apply the changes without restarting the terminal:

```
source ~/.bashrc
```

Then, the next step is to check if the variable has been saved. How to do this? You can type this command in the terminal:

```
echo $LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED
```

```
echo $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
```

The output should look like the image below:

![image](https://github.com/user-attachments/assets/7cacd520-761f-455a-bd47-1b57b1ff1dca)

This means your variable has been saved to your system's environment. Once it's done, you just need to run Label Studio, making sure that the label_studio virtual environment is activated first. After that, you can run Label Studio by typing:

```
label-studio
```

Then, wait until it runs on **localhost**. Once it's open, create a new project by clicking the `Create` button.

![image](https://github.com/user-attachments/assets/ed2db787-bbf3-441f-8dee-aaf6d82f7082)

After that, you can name the project based on your needs.

![image](https://github.com/user-attachments/assets/ee0999bd-63dd-446e-8086-e070a43c3042)

Let's skip the Data Import part and go directly to the Labeling Setup. You can check out the template 👉 [**here**](https://labelstud.io/templates/).

![image](https://github.com/user-attachments/assets/345b6950-9de5-4e28-a59f-b8bf06956077)

You can choose according to your needs. In this example, I'm taking the object detection option.

![image](https://github.com/user-attachments/assets/1ccad72c-b1a2-457e-8edb-d638e4d486a1)

In this section, you can add a label (no. 1) according to your needs and click Add to add a new label (no. 2). Then press `Save`.

![image](https://github.com/user-attachments/assets/60abe6d8-fa2d-4ea8-9cd8-79367e33258a)

Now, the interface should look like the image above, with no data imported yet. To set up local storage, go to `Settings`.

![image](https://github.com/user-attachments/assets/adb7d195-2f79-4364-8c6c-14d2c64e4291)

Then go to Cloud Storage.

![image](https://github.com/user-attachments/assets/3c7e2167-8a0d-43ee-9e70-5fbbd7102496)

Press the **Add Source Storage** button.

![image](https://github.com/user-attachments/assets/457b8024-bd0f-435f-a111-ff1c5fe9e088)

In this section, choose **Local files**.

![image](https://github.com/user-attachments/assets/c3330e2b-37a2-42bf-83cd-3cfdde9d1258)

Now, here's the setup.

![image](https://github.com/user-attachments/assets/c1b6b3e1-749d-4508-a6c7-03261194e267)

1. The **Storage Title** can be filled in freely.

2. In the **Absolute local path**, this should be filled with the path to your dataset, as I explained earlier. Since I previously set the environment variable with the parent path of my dataset as `/home/anggads01/Workspace/Label_Studio/Datasets`, in the **Absolute local path**, I filled it with `/home/anggads01/Workspace/Label_Studio/Datasets/Fruit_Images`, where `Fruit_Images` is the folder containing the images to be labeled.

3. Once that's done, in the **File Filter Regex**, enter the file extension for the files you want to label. Since my data is images, I entered `.*(jpe?g)`. You can adjust this to match the extension of your data. Don't worry, you don't need to be proficient in Regex, just follow the suggestion from the placeholder.

4. Check the box `Treat every bucket object as a source file`.

5. Then click the `Check Connection` button.

6. If successful, you'll see the message `Successfully connected`.

7. Click the Add Storage button, and the interface should look like this:

![image](https://github.com/user-attachments/assets/2908e036-167f-462b-ba1c-a90ceee0fb48)

Then, you'll need to **Sync Storage** so that Label Studio can access your dataset locally. This will take time depending on how much data you have.

![image](https://github.com/user-attachments/assets/3e345b2c-61ed-4d1c-95e0-01e5b7d5c734)

Now you should be able to see that in my dataset, I have **300 images** that need labeling. Then, go back to your project, and now your data should be available.

![image](https://github.com/user-attachments/assets/a0d8f14d-d76c-4160-a5c6-6278070707eb)

For me, the interface looks like this:

![image](https://github.com/user-attachments/assets/85df9342-42f4-4c7c-96c9-067cd85cd367)

And yes… we've successfully completed the **local storage setup** in Label Studio. Woohoo 🎉🎉🎉! I hope you were able to do it too. The next section of my article will be about setting up **Auto Annotation** in Label Studio ✨.

---

Setup Auto Annotation
After setting up local storage, let's dive into a feature that will make your workflow even more efficient: Auto Annotation. Essentially, this feature turns Label Studio into a sort of "smart assistant" that can automatically create annotations based on the machine learning model we've prepared.
Okay, now we're going to go through the detailed steps to set up Auto Annotation using a pretrained model in Label Studio. Get your coffee ready, bro, because this step can get a bit technical, but I'll make sure it's easy to follow. Let's start!
The first step is, of course, to run Label Studio in the terminal. Remember, you need to activate the virtual environment where Label Studio is installed, then run Label Studio by typing:
label-studio
Once it's running, open your browser and access Label Studio through the address that appears in the terminal. By default, it's http://localhost:8080/, just leave it open because we'll set up the project after we successfully set up auto annotation.
Next, you need to download (clone) the label-studio-ml-backend repository. But before that, create a folder where you want to store the repository.
For example, I created a folder named Label_Studio and saved the repo there. To download the repo, open a new tab in Windows Terminal and type the following command in your Ubuntu terminal:
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
It should look something like this:
You can see above that the repository has been cloned. Next, I'm going to give you some tutorials related to the models that can be used for auto annotation in Label Studio. Make sure that after you clone it, you enter the virtual environment that has Label Studio installed. For me:
conda activate label_studio
Setting Up the YOLO Model
Now, navigate to the ML Backend directory:
cd label-studio-ml-backend/label_studio_ml/examples/yolo
After that, you can type:
code .
This means you want to open the yolo folder in Visual Studio Code. Once it's open, find and open the docker-compose.yml file. In this file, there is an important variable: LABEL_STUDIO_API_KEY. You need to fill this with your Label Studio API Key. Here's how:
1. Click on your profile picture at the top right of Label Studio.
2. Click "Account & Settings".
3. Find the API Key in your account menu and copy the value.
Paste it in the LABEL_STUDIO_API_KEY section.
version: "3.8"
services:
    ...
    environment:
      ...
      - LABEL_STUDIO_URL=http://host.docker.internal:8080
      - LABEL_STUDIO_API_KEY=your_api_key_here
      ...
    ...
For the LABEL_STUDIO_URL, just leave it as the default at port 8080, as it's important for connecting the model server to our Label Studio server.
Next, you can open the Dockerfile and pay attention to the following line of code in the Dockerfile:
...
RUN conda install -c "nvidia/label/cuda-12.1.1" cuda -y
ENV CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"
...
Now, you can delete that line. Why? Because it will cause an error since the referenced CUDA version can't be found. I did this based on my trial and error, but don't worry, this doesn't mean we can't use GPU support for inference/prediction. The base image of the Docker already comes with GPU support, as you can see from this line of code:
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
...
The next step is to run the following command to build and run the YOLO container, but make sure Docker Desktop is running. You can go to the Windows menu, search for Docker Desktop, and open it. Wait for it to start, then type the following command in your terminal:
docker-compose up --build
--build is used when we want to build the Docker container for the first time. This usually takes a long time and can be as large as 10 GB, so make sure you have enough Local Storage.
Next, open Label Studio in your browser again, create a project by clicking the Create button, and give your project a name. For example, "YOLO - Bounding Box - Fruits". This is important so you don't forget what dataset you're labeling and what the task is. But basically, name it however you like.
Next, adjust the Labeling Interface. Once the YOLO container is running, you need to adjust the Labeling Interface in Label Studio to match the YOLO format. You can find the guide 👉 here.
Go to the Labeling Setup panel and select Custom template.
After opening the Custom template view, you can follow along with me or adjust it according to your label classes.
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
    <Label value="banana" background="blue" predicted_values="banana"/>
    <Label value="orange" background="yellow" predicted_values="orange"/>
    <Label value="apple" background="green" predicted_values="apple"/>
  </RectangleLabels>
</View>
You can either copy or adjust it by changing the value in the Label tag. The predicted_values section contains the classes that YOLO can predict. You can see the list of classes 👉 here. Next, you'll place the template in the Labeling Interface in Label Studio, as shown below.
Then, click the Save button. If you've already saved it and want to change the color of each label or add a label class, you can go to Settings → Labeling Interface. You can add classes in the add label names section. If you want to change the label color, you can see no. 2, just click on it and adjust it to the color you think fits.
Before we integrate the model into the project, you need to connect your data in local storage to Label Studio. You can check the tutorial for that in the Setup Local Storage in Label Studio section from the previous steps. Now, let's integrate the YOLO model into the project we created. Go to Settings → Model, then click the Connect Model button.
For the configuration, first, give your model a name, then enter the address URL of the model server. By default, it's http://localhost:9090. Don't forget to turn on the Interactive preannotations option. Lastly, click the Validate and Save button, and with this, your model should now be connected to Label Studio.
Now, go to your project, click on the image you want to annotate, and YOLO should automatically detect the objects.
The result should look like this:
Oh, and don't forget to activate both Auto-Annotation and Auto-Accept Suggestions. And yay, we've successfully set up auto annotation in Label Studio! You can explore more because YOLO not only does bounding boxes but also other tasks. You can see 👉 here.
When you're done using the YOLO model, you can shut down the container to save resources. Make sure you're in the yolo directory.
cd label-studio-ml-backend/label_studio_ml/examples/yolo
then run:
docker-compose down
And if you want to run the container again, you can type the following command:
docker-compose up
Don't forget to reconnect the model to Label Studio by going to Settings → Model → Edit → Validate and Save.
Bonus: Using a Custom YOLO Model
Make sure your custom YOLO model is saved in .pt format (this is the standard format for PyTorch models). For example, let's say your model file is named my_custom_model.pt.
Create a Models Folder
Inside the YOLO project folder (where your docker-compose.yml file is located), create a new folder named models if it doesn't already exist. Place your model file (my_custom_model.pt) inside this folder.
yolo/
├── docker-compose.yml
├── models/
    └── my_custom_model.pt
Edit docker-compose.yml File Add volume settings to allow Docker to access your model file. Add the following line (if it's not already there) in the volumes section of the docker-compose.yml file:
services:
  yolo:
    container_name: yolo
    ...
    volumes:
      - ./models:/app/models  # Mapping folder 'models' ke dalam container
    environment:
      - ALLOW_CUSTOM_MODEL_PATH=true  # Aktifkan custom model
    ...
Ensure that the environment variables are correct. You need to configure some environment variables in the docker-compose.yml file:
ALLOW_CUSTOM_MODEL_PATH=true: This gives permission to the Label Studio backend to read the model from a custom path
LABEL_STUDIO_URL: Fill in your Label Studio URL. If everything is running locally, use:

LABEL_STUDIO_URL=http://host.docker.internal:8080
LABEL_STUDIO_API_KEY: You can get this API Key from your Label Studio account (top right corner). Copy the key and place it here.
(Optional) LOG_LEVEL: If you want more detailed logs, you can set this to DEBUG.

environment:
  - ALLOW_CUSTOM_MODEL_PATH=true
  - LABEL_STUDIO_URL=http://host.docker.internal:8080
  - LABEL_STUDIO_API_KEY=your_api_key
  - LOG_LEVEL=DEBUG  # (Opsional) Buat log detail
You need to inform Label Studio that you want to use your custom model for automatic predictions.
In Label Studio, add the model_path parameter in the Labeling Interface when creating your project, for example, for Object Detection:
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels 
          name="label" toName="image" 
          model_path="my_custom_model.pt" 
          model_score_threshold="0.25">
    <Label value="Cat"/>
    <Label value="Dog"/>
  </RectangleLabels>
</View>
Remember, model_path: It should match the model file name and the file must be in the /app/models folder inside the Docker container. Mapping Class Names: Ensure that the names in <Label> match the classes your model predicts. If they don't match, add the predicted_values attribute. For example:
<Label value="Cat" predicted_values="feline"/>
<Label value="Dog" predicted_values="canine"/>
Restart ML Backend
To apply your changes, restart the Docker backend. Make sure you're in the yolo directory.
cd label-studio-ml-backend/label_studio_ml/examples/yolo
docker-compose down
If you made changes to the docker-compose.yml file, you'll need to rebuild by typing the following in your terminal:
docker-compose up --build
If there are no changes, and you just need to place your custom model in the yolo/models folder, you just need to type the following in your terminal:
docker-compose up
Open Label Studio, go to Settings in the project you created, and navigate to the Model tab. Fill in the ML Backend URL (usually http://localhost:9090 if it's running locally).
Add a few tasks (images or other data) to your Label Studio project, and open the task in the Labeling Interface. You should now be able to use your custom model for predictions.
If You Encounter Errors, Check:
Wrong Path: Ensure that the model_path in your labeling configuration exactly matches the model file name inside /app/models.
Label Mismatch: Double-check that your labels in Label Studio match the classes predicted by your model, or use predicted_values to map them.
Model Keypoints and Model Index: If you're using a keypoints model, you need to specify the model_index parameter for each Label tag.

Setup SAM Model
After successfully setting up the YOLO model for a custom task, it's time to level up and dive into another awesome model: SAM (Segment Anything Model). While YOLO is great at detecting objects, SAM is like an artist that can divide an image into areas with incredible detail. The setup process is a bit different, but don't worry, I'll guide you step-by-step to make it smooth.
First, go to the ML Backend directory you previously cloned, but now look for the SAM directory:
cd label-studio-ml-backend/label_studio_ml/examples/segment_anything_model
Then, run the following command:
code .
This step is very similar to setting up the YOLO model. You just need to copy the Label Studio API token into the LABEL_STUDIO_API_KEY variable. Then, for LABEL_STUDIO_HOST, if your Label Studio is running on localhost (usually on port 8080 by default), just copy the URL and add it to the docker-compose.yml file.
version: "3.8"
services:
    ...
    environment:
      ...
      - LABEL_STUDIO_HOST=http://host.docker.internal:8080
      - LABEL_STUDIO_API_KEY=your_api_key_here
      ...
    ...
Copy the following script into your docker-compose.yml file.
version: "3.8"
services:
    segment_anything_model:
        container_name: segment_anything_model
        image: heartexlabs/label-studio-ml-backend:sam-master
        build:
            context: .
            shm_size: '4gb'
            args:
                TEST_ENV: ${TEST_ENV}
        deploy:
            resources:
                limits:
                    memory: 8G
                reservations:
                    memory: 4G
                    devices:
                        - driver: nvidia
                            count: 1
                            capabilities: [gpu]
        environment:
            # specify these parameters if you want to use basic auth for the model server
            - BASIC_AUTH_USER=
            # Change this to your model name: MobileSAM or SAM
            - BASIC_AUTH_PASS=
            - SAM_CHOICE=MobileSAM
            - LOG_LEVEL=DEBUG
            # Enable this to use the GPU
            - NVIDIA_VISIBLE_DEVICES=all
            - WORKERS=1
            - THREADS=8
            # specify the number of workers and threads for the model server
            - MODEL_DIR=/data/models
            # Specify the Label Studio URL and API key to access
            # uploaded, local storage and cloud storage files.
            # Do not use 'localhost' as it does not work within Docker containers.
            # Use prefix 'http://' or 'https://' for the URL always.
            # Determine the actual IP using 'ifconfig' (Linux/Mac) or 'ipconfig' (Windows).
            - LABEL_STUDIO_HOST=http://host.docker.internal:8080
            - LABEL_STUDIO_ACCESS_TOKEN=your_api_key_here
        ports:
            - 9090:9090
        volumes:
            - "./data/server:/data"
Next, open the Dockerfileand pay attention to the following line in the Dockerfile:
# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.8
...
Change the Python version to 3.9 or higher, because if you use version 3.8, you'll get an error related to version compatibility.
The next step is to run the following command to build and run the SAM container, but make sure Docker Desktop is already running. You can go to the Windows menu, search for Docker Desktop, open it, and wait for it to start. Then, run the following command in the terminal:
docker-compose up --build
After that, open Label Studio in your browser, click on the Create button to create a project, and give it a name, such as "SAM - Instance Segmentation - Cars." It's important so you don't forget what dataset you're labeling and what the task is. But, just name it however you feel comfortable.
Next, adjust the Labeling Interface. Once the SAM container is up, you'll need to adjust the Labeling Interface in Label Studio according to SAM's format. You can follow the guide directly 👉 here.
Go to the Labeling Setup panel, and choose Custom template. Once the Custom template page opens, you can follow my instructions or adjust it to match your label classes.
<View>
<Style>
  .main {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 40px 5px 5px 5px;
  }
  .container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
  }
  .column {
    flex: 1;
    padding: 10px;
   margin: 5px; 
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    text-align: center;
  }
  .column .title {
    margin: 0;
    color: #333;
  }
  .column .label {
    margin-top: 10px;
    padding: 10px;
    padding-bottom: 7px; 
    background-color: #f9f9f9;
    border-radius: 3px;
  }
  .lsf-labels {
    margin: 5px 0 0 0; 
  }
  .image-container {
    width: 100%;
    height: 300px;
    background-color: #ddd;
    border-radius: 5px;
  }
</Style>
  
<View className="main">
  <View className="container">
    <View className="column">
      <HyperText value="" name="h1" className="help" inline="true">
        Brush for manual labeling
      </HyperText>
      <View className="label">        
        <BrushLabels name="tag" toName="image">
          <Label value="Foreground" background="#FF0000" />
          <Label value="Background" background="#0d14d3" />
        </BrushLabels>
      </View>
    </View>
    
    <View className="column">
      <HyperText value="" name="h2" className="help" inline="true">
        <span title="1. Click purple auto Keypoints/Rectangle icon on toolbar. 2. Click Foreground/Background label here">
          Keypoints for auto-labeling
        </span>
      </HyperText>
      <View className="label">
        <KeyPointLabels name="tag2" toName="image" smart="true">
          <Label value="Foreground" smart="true" background="#FFaa00" showInline="true" />
          <Label value="Background" smart="true" background="#00aaFF" showInline="true" />
        </KeyPointLabels>
      </View>
    </View>
    
    <View className="column">
      <HyperText value="" name="h3" className="help" inline="true">
        <span title="1. Click purple auto Keypoints/Rectangle icon on toolbar. 2. Click Foreground/Background label here">
          Rectangles for auto-labeling
        </span>
      </HyperText>
      <View className="label">
        <RectangleLabels name="tag3" toName="image" smart="true">
          <Label value="Foreground" background="#FF00FF" showInline="true" />
          <Label value="Background" background="#00FF00" showInline="true" />
        </RectangleLabels>
      </View>
    </View>
    
  </View>
  
  <View className="image-container">
    <Image name="image" value="$image" zoom="true" zoomControl="true" />
  </View>
  
</View>
</View>
You can copy or adjust by changing the value in the Label tag. Then, place the template in the Labeling Interface section in Label Studio as shown below, and click Save.
Before integrating the model into the project, you need to connect your local storage data to Label Studio. You can refer to the Setup Local Storage tutorial in the previous step. Next, integrate the SAM model into the project you created by going to Settings -> Model, then click Connect Model.
For the configuration, first, give your model a name, then fill in the address URL of the model server, which by default is http://localhost:9090. Don't forget to enable the Interactive preannotations option, then click the Validate and Save button. This should connect your model to Label Studio.
Now, go to your project and click on the image you want to annotate. You should now be able to perform segmentation with the help of SAM.
Once you're in the main labeling interface, you need to enable Auto-Annotation.
Enable Auto-Accept Suggestions.
Choose the Auto-detect feature.
Then, you just need to select whether you want to segment using the brush, keypoints, or rectangles mode.

And yay, we have successfully set up auto annotation for the SAM model in Label Studio! If you're done using the SAM container, you can turn it off to save resources. Just make sure you're in the segment_anything_model directory.
cd label-studio-ml-backend/label_studio_ml/examples/segment_anythin_model
Then, run the command:
docker-compose down
And if you want to run the container again, you can type the following command:
docker-compose up
Don't forget to reconnect the model to Label Studio by going to Settings → Model → Edit → Validate and Save.
