---
layout: post
title:  "Linear Regression from Scratch Statistical Approach"
date: 2018-10-08 20:47:28 +0530
categories: Linear Regression 
description : Linear Regression is the process of fitting a line to the dataset. 
---

Links to be Included 
* Link to medium post 
* Link to git hub repo
* Link to source Code

#Comments session
<!--article class="post" -->
			<!--img src ="/assets/site_images/sample.jpg" style="width:128px;height:128px;" alt="Image not found"-->
			<!--h2 title="{{ post.title }}"><a href="{{ post.url }}">{{ post.title }}</a></h2-->
			<!--p> {{post.content | truncatewords: 15}}</p-->
			<!--p>{{ post.date | date_to_string }}</p>  
		<!--/article-->





<ul class="posts">
	<div class="container">
		{% for post in site.posts %}
		<article class="post">
			<h2 title="{{ post.title }}"><a href="{{ post.url }}">{{ post.title }}</a></h2>
			<p> {{post.content | truncatewords: 15}}</p>
			<p>{{ post.date | date_to_string }}</p>  
		</article>
		{% endfor %}
	</div>
</ul>


<p style="float: left;">
					<!--img class="card-img-top" src="/assets/site_images/sample.jpg" alt="Card image cap" style="width:128px;height:128px;"-->
</p>

<div class="card-body">
					<p class="card-text">{{post.description | truncatewords: 15}}</p>
				</div>