---
layout: page
title: Resources
permalink: /resources/
---

## Product Management
Product management produces good decisions about products. Good decisions don't come from nowhere. The chief daily work of a product manager consists of improving the product team's organizational memory, communication, learning from feedback, and direction. It's possible to learn to get better at this.

[Product Management primer](/2019/04/25/product-management-primer.html) by me describing in more detail how to make better decisions, with an example from my time at the Wikimedia Foundation.

#### Writing
The one tool that has historically taught people to remember, communicate, learn, and orient better is writing. Thus learning to write is highly relevant to product management, which is itself often compared to being an editor. Here are some great resources on learning to write:

[*The Sense of Style*](https://stevenpinker.com/publications/sense-style-thinking-persons-guide-writing-21st-century) by Steven Pinker, far superior to the standard book on style Strunk and White. He also gives [talks](https://www.youtube.com/watch?v=3ZKTmsgqi0U) about the high level concepts in this book.

[*Writing Down the Bones*](https://nataliegoldberg.com/books/writing-down-the-bones/) by Natalie Goldberg, preferred by me to the popular The Artist's Way. Natalie's rules of writing practice beginning with "keep your hand moving" are practical and freeing, exactly what you might expect of a buddhist writing coach.

Essays by Paul Graham on writing have helped me a lot. He's sharp and the advice he gives, including any about writing, is solid. Here's a good place to [start](http://www.paulgraham.com/writing44.html).

Jordan Peterson's guide to [writing essays](https://docs.google.com/viewer?url=http://jordanbpeterson.com/wp-content/uploads/2018/02/Essay_Writing_Guide.docx) is well thought out and articulated, starting from why write essays.

## Machine Learning

Learning the field of Machine Learning has been rewarding and challenging. Here are some of the lessons I learned.

**Application**  
The theoretical space of Machine Learning is so broad that it can be easy to get lost. Start by learning more practically. I started with the [Elite Data Science](https://elitedatascience.com/) course. You don't need to know a ton of coding or math to start being able to do some interesting things. And it turns out many of the interesting things you will do, at least in Data Science, have some reasonable definition specifically in the form of [common work flows](https://elitedatascience.com/primer). When you have this framework of application it's easier to learn new concepts. In part this is because you know that you don't have to perfectly understand a tool to apply it, so you will be appropriately patient with learning that new tool.

**Start small**  
There's a reasonable amount of advice online about Machine Learning that tells you to find a project and tackle it, and that tackling that project will teach you as you go along. I did not find this to be helpful advice. A reasonable amount of the projects you will want to tackle will be too big for you. More practical is to learn one technique on one constrained dataset. In my [portfolio](/portfolio/) under projects with preprocessed data you'll see some of mine.

Which techniques? Spend some time [mind mapping](/assets/MachineLearningAlgorithms.png) out the space that you're interested in, this will both give you some direction and it will calm the part of your mind that is worried about what it's not paying attention to. This only works if you make the mind map not if you look at one. Which datasets? the [UCI repository](http://archive.ics.uci.edu/ml/index.php) is good. Here is Jason Brownlee on the [same topic](https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/) more extensively.

Working these smaller projects also allows you to do the work within a day. Partnered with something like the [#100DaysOfCode](https://www.100daysofcode.com/) challenge, this is a great way to learn.

**End to End Projects**  
Of course in the long run you need to learn to deal with data acquisition and data cleaning issues, that's why it's good to have bigger end to end projects as well. It'll be hard to know how big of a project is too big to tackle sometimes. For example, I looked at a [list of potential projects](https://elitedatascience.com/machine-learning-projects-for-beginners#health-care) and found anomaly detection on the Enron corpus of emails to be interesting. When I looked online there was actually surprisingly little research done on the topic. I did find one interesting paper though and decided to [reimplement their findings](/end%20to%20end%20projects/2019/11/24/enron-anomaly-detection.html). It ended up being a 1 month project which required me to teach myself about neural networks and deep learning. This particular project was a bit of a stretch at the time. Course there is nothing like real life projects, and days of debugging, to remind you to be careful each step of the way.   

**Theory**  
Once you've learned enough, it will be hard to be able to remember more information without a theoretical framework to contain and structure the conceptual knowledge you're learning. Algorithms, databases, and computing machines are the tools of Machine Learning. Algorithms are where I needed the most theory to keep advancing my knowledge. The best way to increase theoretical knowledge about algorithms is to implement simple versions of them. Again, Jason Brownlee writes well about [this](https://machinelearningmastery.com/how-to-implement-a-machine-learning-algorithm/). On my [portfolio](/portfolio/) you'll find many implemented algorithms.

**Math**  
To advance your theoretical knowledge you'll eventually need to know some math. The [Deep Learning Book] (http://www.deeplearningbook.org/) has the best material on this that I've seen, the book is clearly written and embedded intelligently into theory and application, so it's not just math for math's sake. Thanks to [i am trask](https://github.com/llSourcell/Learn_Deep_Learning_in_6_Weeks) for the recommendation (and his excellent [6 week deep learning curriculum](https://github.com/llSourcell/Learn_Deep_Learning_in_6_Weeks) adapted from [Siraj Raval](https://www.youtube.com/watch?v=_qjNH1rDLm0&feature=youtu.be)). If you understand the concepts in Part 1 of the Deep Learning Book you'll have a good mathematical framework for understanding the algorithm speak in ML. If you get stuck on any of the linear algebra [3blue1brown](https://www.3blue1brown.com/) is an excellent place to start, along with any necessary reinforcement from [Khan Academy](https://www.khanacademy.org/math/linear-algebra), and Gilbert Strang's [MIT Linear Algebra course](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/).

## Entrepreneurship

Operating companies has forced me to think about what a company is. Thinking about what a company is helps you run companies better. I wrote an essay summarizing some conclusions I've drawn:
[What a company is](/2019/04/25/what-a-company-is.html)

Some management recommendations from the above essay:

[*Mastering the Rockefeller Habits*](https://www.amazon.com/dp/B005J386GS/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1) by Verne Harnish outlines three tasks management must do: establish priorities, look at data, and drive a rhythm of meetings that maintain alignment and accountability. This has formed the basic structure of management for all my teams in the past five years.

[*High Output Management*](https://www.amazon.com/dp/B015VACHOK/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1) by Andrew Grove. No book better breaks down what it is to control the output of a company. He defines management in the most basic manner: to first increase the product of your team, and second to increase the productivity of the teams connected to your team.

[*TopGrading*](https://www.topgrading.com/) by Bradford Smart the book isn't well reviewed but all you need is the interviewing guide. I’ve used a stripped down version of the CIDS interview to both make and attract top hires.  

[Operating startups](/2019/04/25/startup-operations-primer.html) often has similar challenges. After finding a product people want you need to be able to create necessary process, know what to measure, and manage your money.
