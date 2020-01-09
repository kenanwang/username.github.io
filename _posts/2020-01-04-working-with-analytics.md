---
layout: post
title: Understanding Success (working with Analytics)
categories: [Product Management]
tags: [Business]
---

At the Wikimedia Foundation we had a major success in my department during my time there. For years the most pressing issue was a decline in the number of active editors, people who edited 5 or more times in a month. As many of you know, Wikipedia is crowdsourced, so it is these editors who create the articles on Wikipedia. Since the end of 2007 that number had been going down.
![active editors](https://stats.wikimedia.org/EN/PlotEditorsEN.png){: width="100%" style="margin: 20px 0px 10px 0px"}

There were two main theories why why the number of active editors was going down:
1. Between 2004 and 2007 Wikipedia became popular, very popular. Popular enough that vandalism became a thing on Wikipedia, and Wikipedia was important enough that this vandalism had real reach. A journalist named John Seigenthaler had a fake and negative article written about him [for example](https://en.wikipedia.org/wiki/List_of_Wikipedia_controversies#2005). And the incidents only increased in frequency. In response to this Wikipedia editors started to organically create rules and even bureaucracy to stop this kind of abuse of the site. They worked, but they also made it harder to become an editor. Today if you want to edit Wikipedia you need to learn quite a few rules and customs to really feel comfortable on the site.

2. Since 2007 other sites started to pop up, the kinds of sites that might be interesting to the same type of people that are potential editors for Wikipedia: Quora, Stackoverflow, etc. Plus editing Wikipedia had an old school interface.

The idea was that if we could 1. make Wikipedia be more welcoming, both as a community and as a product, and 2. make Wikipedia feel more modern in its interface that we could stop or reverse the decline in active editors.

### Jump in Active Editors metric
The biggest success came when we released a new feature: the ability to make edits from mobile devices. For years you could read Wikipedia on your mobile phone, but you couldn't make edits from there. One of the first features released while I was the product manager at for Wikipedia mobile was mobile editing. Within about a week or two of the release Dario, my analytics lead, excitedly showed me this graph:

![active editors increase](/assets/mobile_contrib_pg1.jpg)

What you'll see is that from one year to the next there was an increase in the number of active editors of 15%. It's a noticeably jump from the date of the release, marked in red. 15% was by far the biggest increase in the number of active editors since late 2007 when it had started dropping. It was exciting and it was definitely time to figure out why exactly this was happening.

### Quantitative analysis
Dario was amazing in providing this analysis. The most useful analyses were:

1. looking at mobile vs. desktop cohorts - analyzing different behavior between two groups of people is not always a simple task, partly this is because it is not trivial to separate two groups of users as it is. Fortunately we had a robust analytics framework in place for this feature well before it was released. Because of this we were able to get quantitative feedback about these editors as well as qualitative. We got an analytical sense that some of these editors were not only making little edits here and there, as we might expect on a mobile phone, but that some where making hundreds of edits. More about the qualitative findings below.

2. looking at mobile user conversion funnels - seeing how a user behaves before, during, and after converting is also not trivial. Users can behave unpredictably, it's important to catch this data and also make sure that all of the relevant data is recorded to the same person. One of the most interesting things we learned quantitatively about these mobile user conversions was that the overwhelming majority of them, 71% stayed editing only on mobile. This was a very interesting finding, partially because on mobile, the users didn't have much interaction with the rules and bureaucracy, the communication tools weren't as robust as on desktop yet. It seemed to support the theory that rules and bureaucracy were a big part of the problem, especially since the early editing interface on mobile was raw.

### Qualitative analysis and user interviews
We learned a lot from the numbers, but looking at the actual edits was equally informative. As I mentioned, I had expected mobile editors to be making small changes here and there, not sweeping changes to articles. I was wrong. Here is an example of a user who basically rewrote an entire article from his phone:

![nowshera article](/assets/mobile_contrib_pg15.jpg)

This was very exciting. It also suggested, perhaps unsurprisingly, that some of these editors were coming from 2nd or 3rd world countries; this was a plus for Wikipedia, making Wikipedia more accessible to a wider audience could only be a plus for its perspectives and its breadth of knowledge.

Lastly, in addition to looking at some of these edits. I messaged some of the top mobile editors. I interviewed them over messages and email. One editor told me how she was using mobile editing to edit more:

> PamD: active on desktop and mobile
> "I enjoy it...it's the sort of thing I'm good at, as a retired librarian... Usually, from a laptop on a desk at home...Often [I edit on mobile] lying in bed early morning, checking watchlist or stub category, finding something I need to edit...[ or] where I am at the time... sitting in the car admiring a view with Mother (96, no short-term memory, not a great conversationalist), or away from home”

She also told me how she could see it improving.
> “Main problem is that it seems to assume that the mobile reader is dumb and doesn't need the same facilities which we provide automatically for a desktop readers...I'm not a pre-teen playing with social media, I'm a mature and experienced editor who sometimes uses a mobile to read and to edit the encyclopedia. And our numbers will be growing, as more people acquire smartphones (even if its their children's or grandchildren's cast-off phones, as in the case of some people I know!)"

Clearly, she thought as I was learning that mobile editing would be more sophisticated than I had originally thought. Many of the features that came later were inspired by comments like this, making more functionality available to mobile users, including complex functions that we wouldn't have prioritized previously.

Full Presentation including analysis of this feature [here](/assets/Mobile_Contributions_Quarterly_Review_10-2013.pdf).

Here's the [New York Times]([New York Times](https://www.nytimes.com/2014/02/10/technology/wikipedia-vs-the-small-screen.html)) on the challenges of mobile on Wikipedia, contrary to how this article is written we were quite optimistic at this time, and for good reason. In large part due to mobile and other changes from our product team Wikipedia has been able to [stop the decline in active editors](https://blog.wikimedia.org/2015/09/25/wikipedia-editor-numbers/).
