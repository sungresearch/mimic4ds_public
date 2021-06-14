/*
- 24h sepsis3
- resources:
    - mimic_derived.sepsis3 (https://github.com/MIT-LCP/mimic-iv/tree/master/concepts/sepsis)
*/

drop table if exists `mimic-iv-ches.cohorts.mimic4ds_sepsis3`;
create table `mimic-iv-ches.cohorts.mimic4ds_sepsis3` as(
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
    -- Exclude stays with sepsis onset <= 4 hours upon ICU admission
    select
      icu.stay_id as stays_to_remove,
    from
      icu_rand icu
      left join `mimic-iv-ches.mimic_derived.sepsis3` s on icu.stay_id = s.stay_id
    where
      s.suspected_infection_time <= datetime_add(icu.intime, interval 4 hour)
    group by
      icu.stay_id
  ),
  sepsis3_label as (
    -- Label = 1 if subject developed sepsis3 between 4 and 168 hours (7 days) after ICU admission
    --      onset of sepsis3 is the time of suspected infection
    -- Label = 0 otherwise
    select
      icu.stay_id,
      suspected_infection_time, sofa_time,
      case
        when 
          suspected_infection_time > datetime_add(icu.intime, interval 4 hour) and
          suspected_infection_time <= datetime_add(icu.intime, interval 168 hour) and
          sepsis3
        then 1
        else 0 
      end as label
    from
      icu_rand icu
      left join `mimic-iv-ches.mimic_derived.sepsis3` s on icu.stay_id = s.stay_id
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
    s.suspected_infection_time, s.sofa_time,
    -- GROUPING
    anchor_year_group as group_var,
    -- LABEL
    case
      when label is null
      then 0
      else label 
    end as label
  from
    icu_rand icu
    left join `mimic-iv-ches.core.admissions` a on icu.hadm_id = a.hadm_id
    left join sepsis3_label s on icu.stay_id = s.stay_id
    left join excl_stays e on icu.stay_id=e.stays_to_remove
  where
    stays_to_remove is null and
    rn = 1
  );
select * from `mimic-iv-ches.cohorts.mimic4ds_sepsis3`  