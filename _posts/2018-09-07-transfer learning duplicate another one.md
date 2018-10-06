---
layout: post
title: "Transfer Learning for Newbies duplicate"
date: 2018-09-07
---
another one
Treansfer Learning technique is used to indentify tom and jerry.

<meta http-equiv=Content-Type content="text/html; charset=windows-1252">
<meta name=Generator content="Microsoft Word 15 (filtered)">
<style>
<!--
 /* Font Definitions */
 @font-face
	{font-family:"Cambria Math";
	panose-1:2 4 5 3 5 4 6 3 2 4;}
@font-face
	{font-family:Calibri;
	panose-1:2 15 5 2 2 2 4 3 2 4;}
@font-face
	{font-family:"Noto Sans Symbols";}
@font-face
	{font-family:Georgia;
	panose-1:2 4 5 2 5 4 5 2 3 3;}
@font-face
	{font-family:Cambria;
	panose-1:2 4 5 3 5 4 6 3 2 4;}
@font-face
	{font-family:Consolas;
	panose-1:2 11 6 9 2 2 4 3 2 4;}
@font-face
	{font-family:"Helvetica Neue";}
 /* Style Definitions */
 p.MsoNormal, li.MsoNormal, div.MsoNormal
	{margin-top:0in;
	margin-right:0in;
	margin-bottom:8.0pt;
	margin-left:0in;
	line-height:107%;
	font-size:11.0pt;
	font-family:"Calibri",sans-serif;}
h1
	{margin-top:24.0pt;
	margin-right:0in;
	margin-bottom:6.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:24.0pt;
	font-family:"Calibri",sans-serif;}
h2
	{margin-top:.25in;
	margin-right:0in;
	margin-bottom:4.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:18.0pt;
	font-family:"Calibri",sans-serif;}
h3
	{margin-top:14.0pt;
	margin-right:0in;
	margin-bottom:4.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:14.0pt;
	font-family:"Calibri",sans-serif;}
h4
	{margin-top:12.0pt;
	margin-right:0in;
	margin-bottom:2.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:12.0pt;
	font-family:"Calibri",sans-serif;}
h5
	{margin-top:11.0pt;
	margin-right:0in;
	margin-bottom:2.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:11.0pt;
	font-family:"Calibri",sans-serif;}
h6
	{margin-top:10.0pt;
	margin-right:0in;
	margin-bottom:2.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:10.0pt;
	font-family:"Calibri",sans-serif;}
p.MsoTitle, li.MsoTitle, div.MsoTitle
	{margin-top:24.0pt;
	margin-right:0in;
	margin-bottom:6.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:36.0pt;
	font-family:"Calibri",sans-serif;
	font-weight:bold;}
p.MsoSubtitle, li.MsoSubtitle, div.MsoSubtitle
	{margin-top:.25in;
	margin-right:0in;
	margin-bottom:4.0pt;
	margin-left:0in;
	line-height:107%;
	page-break-after:avoid;
	font-size:24.0pt;
	font-family:"Georgia",serif;
	color:#666666;
	font-style:italic;}
.MsoChpDefault
	{font-family:"Calibri",sans-serif;}
.MsoPapDefault
	{margin-bottom:8.0pt;
	line-height:107%;}
@page WordSection1
	{size:8.5in 11.0in;
	margin:1.0in 1.0in 1.0in 1.0in;}
div.WordSection1
	{page:WordSection1;}
 /* List Definitions */
 ol
	{margin-bottom:0in;}
ul
	{margin-bottom:0in;}
-->
</style>




<div class=WordSection1>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:24.0pt;
line-height:107%'>Linear Regression using Tensorflow Estimator</span></b></p>

<p class=MsoNormal style='text-align:justify'><img width=624 height=430
id=image4.png
src="2018-09-11-Linear-Regression-using-Tensorflow-Estimator_files/image001.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>The Theory</span></b></p>

<p class=MsoNormal>Linear Regression is the process of fitting a line to the
dataset.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Single
Variable Linear Regression</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>The
Mathematics</span></b></p>

<p class=MsoNormal>The equation of Line is</p>

<p class=MsoNormal align=center style='text-align:center'><span
style='font-size:11.0pt;line-height:107%;font-family:"Calibri",sans-serif'><img
width=87 height=19
src="2018-09-11-Linear-Regression-using-Tensorflow-Estimator_files/image002.gif"></span></p>

<p class=MsoNormal>Where,</p>

<p class=MsoNormal> y = dependent variable</p>

<p class=MsoNormal> X = independent variable</p>

<p class=MsoNormal>C = intercept </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>The algorithm is trying to fit a line to the data by
adjusting the values of m and c. Its Objective is to attain to a value of m
such that for any given value of x it would be properly predicting the value of
y.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>There are various ways in which we can attain the values of
m and c </p>

<ol style='margin-top:0in' start=1 type=1>
 <li class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
     115%;border:none'><span style='color:black'>Statistical approach</span></li>
 <li class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
     115%;border:none'><span style='color:black'>Iterative approach</span></li>
</ol>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>Here we are using a scikit learn framework which internally
uses iterative approach to attain the linear regression </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>The
Dataset</span></b><span style='font-size:18.0pt;line-height:107%'> </span></p>

<p class=MsoNormal>Dataset consists of two columns namely X and y</p>

<p class=MsoNormal>Where</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>For <span style='background:white'>List Price Vs. Best Price
for a New GMC Pickup</span> dataset</p>

<p class=MsoNormal>X = <span style='background:white'>List price (in $1000) for
a GMC pickup truck</span></p>

<p class=MsoNormal>Y = <span style='background:white'>Best price (in $1000) for
a GMC pickup truck</span></p>

<p class=MsoNormal>The data is taken from <i><span style='background:white'>Consumer’s
Digest.</span></i></p>

<p class=MsoNormal><i><span style='background:white'>&nbsp;</span></i></p>

<p class=MsoNormal>For Fire and Theft in Chicago </p>

<p class=MsoNormal>X = fires per 100 housing units</p>

<p class=MsoNormal>Y = thefts per 1000 population within the same Zip code in
the Chicago metro area</p>

<p class=MsoNormal>The data is taken from U.S Commission of Civil Rights.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>For Auto Insurance in Sweden dataset</p>

<p class=MsoNormal>X = number of claims</p>

<p class=MsoNormal>Y = total payment for all the claims in thousands of Swedish
Kronor</p>

<p class=MsoNormal>The data is taken from Swedish Committee on Analysis of Risk
Premium in Motor Insurance.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>For Gray Kangaroos dataset</p>

<p class=MsoNormal>X = <span style='background:white'>nasal length (mm ¥10)</span></p>

<p class=MsoNormal>Y = <span style='background:white'>nasal width (mm ¥ 10)</span><br>
<span style='background:white'>for a male gray kangaroo from a random sample of
such animals</span></p>

<p class=MsoNormal>The data is taken from <span style='background:white'>Australian<i>
Journal of Zoology</i>, Vol. 28, p607-613.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><u><span style='color:blue'><a
href="http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html"><span
style='color:blue'>Link to All Datasets</span></a></span></u></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>&nbsp;</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>The Code</span></b></p>

<p class=MsoNormal style='text-align:justify'>The Code was written in three
phases</p>

<ol style='margin-top:0in' start=1 type=1>
 <li class=MsoNormalCxSpMiddle style='margin-bottom:0in;margin-bottom:.0001pt;
     text-align:justify;line-height:115%'>Data preprocessing phase</li>
 <li class=MsoNormalCxSpMiddle style='margin-bottom:0in;margin-bottom:.0001pt;
     text-align:justify;line-height:115%'>Training</li>
 <li class=MsoNormalCxSpMiddle style='margin-bottom:0in;margin-bottom:.0001pt;
     text-align:justify;line-height:115%'>Prediction and plotting</li>
</ol>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Data
Preprocessing Phase</span></b></p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Imports</span></b></p>

<p class=MsoNormal>Numpy import for array processing, python doesn’t have built
in array support. The feature of working with native arrays can be used in
python with the help of numpy library.</p>

<p class=MsoNormal>Pandas is a library of python used for working with tables,
on importing the data, mostly data will be of table format, for ease
manipulation of tables pandas library is imported</p>

<p class=MsoNormal>Matplotlib is a library of python used to plot graphs, for
the purpose of visualizing the results we would be plotting the results with
the help of matplotlib library.</p>

<p class=MsoNormal>Tensorflow import since we are going to use tensorflow
framework for building model.</p>

<p class=MsoNormal>&nbsp;</p>

<table class=a border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Imports</span><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'><br>
  </span><b><span style='font-family:Consolas;color:#8BE9FD;background:#282A36'>import</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> numpy </span><b><span
  style='font-family:Consolas;color:#8BE9FD;background:#282A36'>as</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> np<br>
  </span><b><span style='font-family:Consolas;color:#8BE9FD;background:#282A36'>import</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> pandas </span><b><span
  style='font-family:Consolas;color:#8BE9FD;background:#282A36'>as</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> pd<br>
  </span><b><span style='font-family:Consolas;color:#8BE9FD;background:#282A36'>import</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>
  matplotlib.pyplot </span><b><span style='font-family:Consolas;color:#8BE9FD;
  background:#282A36'>as</span></b><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'> plt<br>
  </span><b><span style='font-family:Consolas;color:#8BE9FD;background:#282A36'>import</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> tensorflow </span><b><span
  style='font-family:Consolas;color:#8BE9FD;background:#282A36'>as</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> tf</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Reading
the dataset from data</span></b></p>

<p class=MsoNormal>In this line of code using the read_excel method of pandas
library, the dataset has been imported from data folder and stored in dataset
variable.</p>

<p class=MsoNormal>On visualizing the dataset, it contains of two columns X and
Y where X is dependent variable and Y is Independent Variable.</p>

<p class=MsoNormal>Note : On using Grey Kangaroos dataset,the data is
normalised, standardised,having a lot of inbuilt variance and outliers the code
would result in a gradient exploding problem.</p>

<table class=a0 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Reading the dataset from data</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  dataset = pd.read_csv(</span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>r'..\\data\\prices.csv'</span><span style='font-family:
  Consolas;color:#F8F8F2;background:#282A36'>)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><span style='font-size:10.5pt;line-height:107%;font-family:
"Helvetica Neue";color:black;background:white'>On viewing the dataset, it
contains of two columns X and Y where X is dependent variable and Y is
Independent Variable.</span></p>

<p class=MsoNormal><img border=0 width=183 height=139 id=image7.png
src="2018-09-11-Linear-Regression-using-Tensorflow-Estimator_files/image003.jpg"></p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Creating
Dependent and Independent variables</span></b></p>

<p class=MsoNormal>The X Column from the dataset is extracted into an X
variable of type numpy, similarly the y variable</p>

<p class=MsoNormal>X is an independent variable </p>

<p class=MsoNormal>Y is dependent variable Inference</p>

<table class=a1 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Creating Dependent and Independent variables</span><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  X = dataset[</span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'X'</span><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'>].values<br>
  y = dataset[</span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'Y'</span><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'>].values</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal><img width=256 height=87
src="2018-09-11-Linear-Regression-using-Tensorflow-Estimator_files/image004.gif"
align=left hspace=12 vspace=12></p>

<p class=MsoNormal>On input 10 it would result in a pandas Series object</p>

<p class=MsoNormal>So, values attribute is used to attain an numpy array</p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Visualizing
the data </span></b></p>

<p class=MsoNormal>The step is to just see how the dataset is </p>

<p class=MsoNormal>On visualization the data would appear something like this</p>

<p class=MsoNormal>The X and Y attributes would vary based on dataset.</p>

<table class=a2 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Visualizing the data </span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  title=</span><span style='font-family:Consolas;color:#F1FA8C;background:#282A36'>'Linear
  Regression on &lt;Dataset&gt;'</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  x_axis_label = </span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'X-value &lt; The corresponding attribute of X in dataset
  &gt;'</span><span style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  y_axis_label = </span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'y-value &lt; The corresponding attribute of X in dataset
  &gt;'</span><span style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  plt.scatter(X,y)<br>
  plt.title(title)<br>
  plt.xlabel(x_axis_label)<br>
  plt.ylabel(y_axis_label)<br>
  plt.show()</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img border=0 width=515 height=371 id=image10.png
src="2018-09-11-Linear-Regression-using-Tensorflow-Estimator_files/image005.jpg"></p>

<p class=MsoNormal><b><span style='font-size:18.0pt;line-height:107%'>Splitting
the data into training set and test set</span></b></p>

<p class=MsoNormal>We are splitting the whole dataset into training and test
set where training set is used for fitting the line to data and test set is
used to check how good the line if for the data.</p>

<table class=a3 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Splitting the data into training set and test set</span><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.8)])<br>
  y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.8)])</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>&nbsp;</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Training Phase</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Variables for training </span></b></p>

<p class=MsoNormalCxSpMiddle style='margin-top:0in;margin-right:0in;margin-bottom:
0in;margin-left:.5in;margin-bottom:.0001pt;text-align:justify;text-indent:-.25in;
border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='color:black'>Epochs: stands for how many time the
whole data is put through on forward propagation and one backward propagation.</span></p>

<p class=MsoNormalCxSpMiddle style='margin-left:.5in;text-align:justify;
text-indent:-.25in;border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='color:black'>Learning Rate: is a hyperparameter in
backpropagation algorithm to adjust the variables in graph based on loss </span>obtained<span
style='color:black'> in forward propagation</span></p>

<p class=MsoNormalCxSpMiddle style='margin-left:.5in;text-align:justify;
text-indent:-.25in;border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span>&nbsp;</p>

<table class=a4 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Variables </span><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'><br>
  epochs = 100<br>
  learning_rate = 0.001</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Feature Columns</span></b></p>

<p class=MsoNormal style='text-align:justify'>These are the features or
Independent variables used for training. We are transforming the numpy arrays
into tensorflow understandable feature columns specifying the column name as
key. This feature column would be fed into tensorflow estimators.</p>

<table class=a5 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Feature Columns</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  feature_columns = [tf.feature_column.numeric_column(key=</span><span
  style='font-family:Consolas;color:#F1FA8C;background:#282A36'>&quot;X&quot;</span><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>)]</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Creating feature dictionaries</span></b></p>

<p class=MsoNormal style='text-align:justify'>These dictionaries are used in
creating in the input function to model.train  and model.predict</p>

<p class=MsoNormalCxSpMiddle style='margin-top:0in;margin-right:0in;margin-bottom:
0in;margin-left:.5in;margin-bottom:.0001pt;text-align:justify;text-indent:-.25in;
border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='color:black'>features_train: used in input function
of model.train</span></p>

<p class=MsoNormalCxSpMiddle style='margin-left:.5in;text-align:justify;
text-indent:-.25in;border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='color:black'>features_test: used in input function
of model.predict</span></p>

<p class=MsoNormalCxSpMiddle style='margin-left:.5in;text-align:justify;
border:none'>&nbsp;</p>

<table class=a6 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Creating feature dictionaries</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  features_train = {</span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'X'</span><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'>:X_train}<br>
  features_test  = {</span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'X'</span><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'>:X_test}</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>&nbsp;</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Creating an Input function which would return a batch dataset
on every call</span></b></p>

<p class=MsoNormal style='text-align:justify'>The input functions are written
for the tensorflow estimator function. The estimator would be expecting a batch
dataset of which would return a tuple of features and labels.</p>

<p class=MsoNormal style='text-align:justify'>The type of processing expected
is </p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<table class=a7 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><b><span style='font-family:Consolas;color:#FF79C6;
  background:#282A36'>def</span></b><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'> </span><b><span style='font-family:Consolas;
  color:#F1FA8C;background:#282A36'>train_input_fn</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>(features,
  labels, batch_size):<br>
      </span><span style='font-family:Consolas;color:#F1FA8C;background:#282A36'>&quot;&quot;&quot;An
  input function for training&quot;&quot;&quot;</span><span style='font-family:
  Consolas;color:#F8F8F2;background:#282A36'><br>
      </span><span style='font-family:Consolas;color:#6272A4;background:#282A36'>#
  Convert the inputs to a Dataset.</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
      dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))<br>
      </span><span style='font-family:Consolas;color:#6272A4;background:#282A36'>#
  Shuffle, repeat, and batch the examples.</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
      </span><b><span style='font-family:Consolas;color:#8BE9FD;background:
  #282A36'>return</span></b><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'> dataset.shuffle(1000).repeat().batch(batch_size)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>It would return a batch
tf.data.Dataset Object</p>

<p class=MsoNormal style='text-align:justify'>Another acceptable format of
input function is </p>

<table class=a8 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><b><span style='font-family:Consolas;color:#FF79C6;
  background:#282A36'>def</span></b><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'> </span><b><span style='font-family:Consolas;
  color:#F1FA8C;background:#282A36'>input_evaluation_set</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>():<br>
      features = {</span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'SepalLength'</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'>: np.array([6.4, 5.0]),<br>
                  </span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'SepalWidth'</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'>:  np.array([2.8, 2.3]),<br>
                  </span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'PetalLength'</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'>: np.array([5.6, 3.3]),<br>
                  </span><span style='font-family:Consolas;color:#F1FA8C;
  background:#282A36'>'PetalWidth'</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'>:  np.array([2.2, 1.0])}<br>
      labels = np.array([2, 1])<br>
      </span><b><span style='font-family:Consolas;color:#8BE9FD;background:
  #282A36'>return</span></b><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'> features, labels</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>It would return a tuple of two
elements, first element features dict and second element labels</p>

<p class=MsoNormal style='text-align:justify'>Other functions which would
support input format are numpy_input_fn and pandas_input_fn</p>

<p class=MsoNormal style='text-align:justify'>For more docs and reference</p>

<p class=MsoNormalCxSpMiddle style='margin-top:0in;margin-right:0in;margin-bottom:
0in;margin-left:.5in;margin-bottom:.0001pt;text-align:justify;text-indent:-.25in;
border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><u><span style='color:#0563C1'><a
href="https://www.tensorflow.org/guide/premade_estimators"><span
style='color:#0563C1'>Tensorflow Premade Estimator Input Functions</span></a></span></u></p>

<p class=MsoNormalCxSpMiddle style='margin-left:.5in;text-align:justify;
text-indent:-.25in;border:none'><span style='font-family:"Noto Sans Symbols"'>&#9679;<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='color:black'> </span><u><span style='color:#0563C1'><a
href="https://www.tensorflow.org/api_docs/python/tf/estimator/inputs"><span
style='color:#0563C1'>Estimator Inputs Module</span></a></span></u></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<table class=a9 border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Creating an Input function which would return a batch dataset on
  every call</span><span style='font-family:Consolas;color:#F8F8F2;background:
  #282A36'><br>
  </span><b><span style='font-family:Consolas;color:#FF79C6;background:#282A36'>def</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> </span><b><span
  style='font-family:Consolas;color:#F1FA8C;background:#282A36'>input_function</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>(features,
  labels, batch_size):<br>
      data = tf.data.Dataset.from_tensor_slices((dict(features), labels))    <br>
      </span><b><span style='font-family:Consolas;color:#8BE9FD;background:
  #282A36'>return</span></b><span style='font-family:Consolas;color:#F8F8F2;
  background:#282A36'>
  (data.shuffle(10).batch(5).repeat().make_one_shot_iterator().get_next()) </span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Making the lambda function of train dataset</span></b></p>

<p class=MsoNormal style='text-align:justify'>Estimator would be expecting
lambda function without any arguments</p>

<table class=aa border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Making the lambda function of train dataset</span><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  input_train = </span><b><span style='font-family:Consolas;color:#8BE9FD;
  background:#282A36'>lambda</span></b><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'>: input_function(features_train, y_train,5)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Build the Estimator</span></b></p>

<p class=MsoNormal style='text-align:justify'>Tensorflow premade estimator are
high level api. These estimators provide a very high level implementation of
machine learning models. Here in the code we are using the LinearRegressor
class</p>

<table class=ab border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Build the Estimator.</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  model = tf.estimator.LinearRegressor(feature_columns=feature_columns)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Train the model</span></b></p>

<p class=MsoNormal style='text-align:justify'>Training is the process of tuning
the models parameters with the provided input data. model.train would take care
of calling the input_train which would feed the model with input data of
shuffled batches. The model would be trained for the given number of epochs.</p>

<table class=ac border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Train the model.</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  model.train(input_fn = input_train, steps = epochs)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Creating an input function for prediction</span></b></p>

<p class=MsoNormal style='text-align:justify'>Similar to train input function
predict input function is also create using pre-built `tf.estimator.input`
module.</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<table class=ad border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Creating a input function for prediction</span><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(features_test, shuffle=</span><b><span
  style='font-family:Consolas;color:#8BE9FD;background:#282A36'>False</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>&nbsp;</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Extracting the y-predicted values into a numpy array</span></b></p>

<p class=MsoNormal style='text-align:justify'>Converting the values in the
generator to numpy array for the ease of plotting.</p>

<p class=MsoNormal style='text-align:justify'><a name="_gjdgxs"></a>creating a
list -&gt; iterating over the generator and appending values -&gt; converting
the list to numpy array</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<table class=ae border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Extracting the y-predicted values into a numpy array</span><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'><br>
  y_predicted = []<br>
  </span><b><span style='font-family:Consolas;color:#8BE9FD;background:#282A36'>for</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'> prediction </span><b><span
  style='font-family:Consolas;color:#8BE9FD;background:#282A36'>in</span></b><span
  style='font-family:Consolas;color:#F8F8F2;background:#282A36'>
  predict_results:<br>
      y_predicted.append(prediction[</span><span style='font-family:Consolas;
  color:#F1FA8C;background:#282A36'>'predictions'</span><span style='font-family:
  Consolas;color:#F8F8F2;background:#282A36'>])<br>
  y_predicted = np.array(y_predicted)</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt;
line-height:107%'>Visualizing the Results</span></b></p>

<p class=MsoNormal style='text-align:justify'>As we have predicted the y-values
for a set of x-values we are visualizing the results to check how good did our
line fit for our predictions.</p>

<p class=MsoNormal style='text-align:justify'>The plot shows the red points are
the data points are actual values where the blue line is the predictions</p>

<table class=af border=0 cellspacing=0 cellpadding=0 style='margin-left:5.0pt;
 border-collapse:collapse'>
 <tr>
  <td width=624 valign=top style='width:6.5in;background:#282A36;padding:5.0pt 5.0pt 5.0pt 5.0pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  115%;border:none'><span style='font-family:Consolas;color:#6272A4;background:
  #282A36'># Visualizing the Results</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'><br>
  plt.scatter(X_test,y_test,c=</span><span style='font-family:Consolas;
  color:#F1FA8C;background:#282A36'>'red'</span><span style='font-family:Consolas;
  color:#F8F8F2;background:#282A36'>)<br>
  plt.plot(X_test,y_predicted,c=</span><span style='font-family:Consolas;
  color:#F1FA8C;background:#282A36'>'green'</span><span style='font-family:
  Consolas;color:#F8F8F2;background:#282A36'>)<br>
  plt.title(title)<br>
  plt.xlabel(x_axis_label)<br>
  plt.ylabel(y_axis_label)<br>
  plt.show()</span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><img border=0 width=515
height=371 id=image8.png
src="2018-09-11-Linear-Regression-using-Tensorflow-Estimator_files/image006.jpg"></p>

</div>




