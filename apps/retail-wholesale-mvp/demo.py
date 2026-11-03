"""Illustrative, generated aggregate demand. No customers or real sales records."""
import math

def sample_payload():
    cfg=[
      ("WATER-FILTER",42,320,38,6,220,100,0.30,.85,11),
      ("SOLAR-LAMP",18,180,12,4,430,260,0.45,.72,17),
      ("CLEANING-KIT",64,570,54,5,145,92,0.2,.92,23),
    ]
    rows=[]
    for sku,base,stock,unit_cost,lead,retail,wholesale,share,reach,seed in cfg:
        daily=[]
        for day in range(35):
            weekly=1+0.16*math.sin(2*math.pi*day/7)
            drift=1+0.002*day
            noise=(((day*19+seed*17)%11)-5)*0.55
            daily.append(max(0,round(base*weekly*drift+noise)))
        rows.append(dict(sku=sku,daily_sales=daily,on_hand=stock,reserved=8,incoming=0,
                         unit_cost_zar=unit_cost,retail_price_zar=retail,
                         wholesale_price_zar=wholesale,wholesale_fraction=share,
                         lead_days=lead,review_days=7,stockout_days=[],
                         promotion_zar=0,place_reach=reach))
    return {"products":rows,"cash_available_zar":55000,
            "production_capacity_units":800,"collection_fraction":0.8}
