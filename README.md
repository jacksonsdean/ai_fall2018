# Final project for Introduction to Artificial Intelligence
COMP 484 with Susan Fox, Fall 2018
------------------------------------------------------------

Set up git:
<ul>
<li>in directory, run $<code>git branch --set-upstream-to=origin/master </code> </li>
	
<li>to push, add the files then <code>git commit -m "some useful comment about the commit"</code> then <code>git pull --rebase && git push</code></li>
</ul>

In order to work with this code, use bash and source bin/activate:
<ul>
<li>$<code>source bin/activate</code></li>
<li>You should see "(final_project)" before your bash prompt</li>
<li>To leave the virtual environment when you're done working with the code, type <code>deactivate</code></li>
</ul>


Now update pip and install dependencies:
<ul>
  <li>(with virtual environment sourced)$<code>pip install --upgrade pip && pip install -r requirements.txt</code></li>
</ul>

The dataset is located at: http://marsyasweb.appspot.com/download/data_sets/
The data processing python file expects the files to be in a folder called "data" with ten subfolders, one for each genre (IE the first jazz song's path is "./data/jazz/jazz.00000.au").


<b>Note:</b>

To run on windows with python 3.7.x, run as administrator, use "py -m pip install XXXXX" to install packages and "py XXX.py" to run files

