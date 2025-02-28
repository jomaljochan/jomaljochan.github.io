---
title: London Housing Market
date: 2024-11-01 08:00:00 - 0000
categories: [London, Real Estate, Data Science, Analytics]
tags: [london, realestate, datascience, analytics]
#image: /path/to/image
alt: "London Housing Market"
---

# Covid Impact Analysis

## **Table of Contents**
* TOC
{:toc}

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


<div class='tableauPlaceholder' id='viz1739716997971' style='position: relative'>
    <noscript>
        <a href='#'>
            <img alt='London housing market' 
                 src='https://public.tableau.com/static/images/Lo/LondonHousing_16700715459660/Story1/1_rss.png' 
                 style='border: none' />
        </a>
    </noscript>
    <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https://public.tableau.com/' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='LondonHousing_16700715459660/Story1' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/Lo/LondonHousing_16700715459660/Story1/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-GB' />
        <param name='filter' value='publish=yes' />
    </object>
</div>

<script type='text/javascript'>
    var divElement = document.getElementById('viz1739716997971');
    var vizElement = divElement.getElementsByTagName('object')[0];
    vizElement.style.width = '1000px';
    vizElement.style.height = (divElement.offsetWidth * 0.75) + 'px';
    
    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>

## Conclusion