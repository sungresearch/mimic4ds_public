/*
- 24h invasive ventilation (tracheostomy, endotracheal tube, or mechanical ventilation)
- resources:
    - mimic_derived.ventilation (https://github.com/MIT-LCP/mimic-iv/blob/master/concepts/treatment/ventilation.sql)
*/

drop table if exists `mimic-iv-ches.cohorts.mimic4ds_vent`;
create table `mimic-iv-ches.cohorts.mimic4ds_vent` as(
  with icu_rand as (
    -- select a random ICU stay for each patient from anchor_year
        select i.subject_id, intime, outtime, i.hadm_id, i.stay_id,
            p.anchor_year_group,
            -- pseudo-random selection using a hash function (farm_fingerprint) to hash columns 
            -- to a deterministic hash, and then use the hash as the "seeded random number"
            row_number() over (partition by i.subject_id 
                                order by farm_fingerprint(concat(cast(i.subject_id as string),cast(i.intime as string),cast(i.outtime as string)))) as rn 
        from 
            `mimic-iv-ches.icu.icustays` i
            inner join `mimic-iv-ches.core.patients` p on i.subject_id = p.subject_id
        where 
            extract(year from i.intime) - anchor_year = 0 and
            datetime_diff(outtime,intime,hour)>4
  ),
  excl_stays as (
    -- Exclude stays with invasive ventilation onset <= 4 hours upon admission
    select
      icu.stay_id as stays_to_remove
    from
      icu_rand icu
      left join `mimic-iv-ches.mimic_derived.ventilation` v on icu.stay_id = v.stay_id
    where
      v.starttime <= datetime_add(icu.intime, interval 4 hour) and
      v.ventilation_status in ("InvasiveVent", "Trach")
    group by
      icu.stay_id
  ),
  vent_label as (
    -- Label = 1 If subject was intubated (Mechanical Vent & Tracheostomy) between 
    --    4 & 28 hours after ICU admission (i.e., over the 24 hours after being in 
    --    the ICU for 4 hours) 
    -- Label = 0 Otherwise
    select
      icu.stay_id,
      max(
        case
          when
            v.ventilation_status in ("InvasiveVent","Trach")
          then 1
          else 0 end
      ) as label
    from 
      icu_rand icu
      left join `mimic-iv-ches.mimic_derived.ventilation` v on icu.stay_id = v.stay_id
    where
      v.starttime > datetime_add(icu.intime, interval 4 hour) and
      v.starttime <= datetime_add(icu.intime, interval 28 hour)
    group by
      icu.stay_id
  )
  select
    -- IDs
    icu.subject_id,
    icu.hadm_id,
    icu.stay_id,
    -- TIME
    icu.intime as ref_datetime,
    icu.intime, icu.outtime,
    a.admittime, a.dischtime,
    -- GROUPING
    anchor_year_group as group_var,
    -- LABEL
    case 
      when 
        label is null
      then 0
      else label end as label
  from
    icu_rand icu
    left join `mimic-iv-ches.core.admissions` a on icu.hadm_id = a.hadm_id
    left join vent_label v on icu.stay_id = v.stay_id
    left join excl_stays e on icu.stay_id= e.stays_to_remove
  where
    stays_to_remove is null and
    rn = 1
);
select * from `mimic-iv-ches.cohorts.mimic4ds_vent`
