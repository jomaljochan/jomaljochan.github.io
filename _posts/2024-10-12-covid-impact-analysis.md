---
title: Covid Impact Analysis
date: 2024-10-12 08:00:00 - 0000
categories: [Data Science, Analytics, AI, Marketing, Branding]
tags: [datascience, analytics, marketing, branding]
#image: /path/to/image
alt: "Covid Impact Analysis"
---

# Covid Impact Analysis

## Story

London housing prices have been rising sharply since the last few years. But does this trend
hold all across London? Or is it a very regional phenomenon? When we analyse the London
housing sale data from 2000 to 2021, we can see the prices go up sharply in the Westminster
and Kensington & Chelsea districts in particular, when compared to Greenwich for example as a
control group.

- From figure 1, the map shows that these areas have overtly high median house prices
when compared to Greenwich. Let's analyse this further.

- To understand these differences, in figure 2, we look into the price variation across the
years, comparing between old and new buildings. This shows that post 2010, the prices
of new houses have constantly increased much past reasonable rates. The old house
prices have also seen a similar trend post 2010, but the prices have stabilised post
2014. Moreover, we see a significant difference in the price hike trend between higher
priced and lower priced areas post 2010.

- Figure 3 shows that the sales number of the old houses have gone down as overall
demand has gone down. The sale numbers of new houses have been historically low
due to the central nature of the area. Also, we can see there is no difference between
the higher priced and lower priced areas in actual sales numbers.

- Finally, in figure 4, we also look at the CO2 emissions of these new and old houses to
find that the CO2 emissions of the new houses are much lesser than that of the old
houses.

We can conclude that the newer houses in the busy central London area, although low in sales
numbers, have the highest prices, but this also comes with a low carbon footprint. Nevertheless,
the prices of these newer houses in the central area are increasing at a higher rate than say
Greenwich, even when the sales numbers and CO2 emissions are similar in both areas: the
Central London areas are overpriced when compared to other areas although the value
proposition seems to be similar.

## Dashboard

<script type='text/javascript' src='https://public.tableau.com/javascripts/api/viz_v1.js'></script>
<div class='tableauPlaceholder' id='viz1739716176443'>
    <noscript>
        <a href='#'>
            <img alt='Dashboard 1' src='https://public.tableau.com/static/images/Co/Coviddata_16700700894990/Dashboard1/1_rss.png' style='border: none' />
        </a>
    </noscript>
    <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='Coviddata_16700700894990/Dashboard1' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/Co/Coviddata_16700700894990/Dashboard1/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
    </object>
</div>

<script type='text/javascript'>
    var divElement = document.getElementById('viz1739716176443');
    var vizElement = divElement.getElementsByTagName('object')[0];
    if (divElement.offsetWidth > 800) {
        vizElement.style.width='1000px';
        vizElement.style.height='827px';
    } else if (divElement.offsetWidth > 500) {
        vizElement.style.width='1000px';
        vizElement.style.height='827px';
    } else {
        vizElement.style.width='100%';
        vizElement.style.height='1677px';
    }
</script>

## Conclusion