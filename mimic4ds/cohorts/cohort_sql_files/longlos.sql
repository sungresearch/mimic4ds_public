drop table if exists `mimic-iv-ches.cohorts.mimic4ds_longlos`;
create table `mimic-iv-ches.cohorts.mimic4ds_longlos` as(
    with icu_last as (
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
    )
    select
        -- IDs
        i.subject_id, i.hadm_id, i.stay_id,
        -- TIME
        i.intime, i.outtime, a.admittime, a.dischtime,
        i.intime as ref_datetime,
        -- GROUPING
        anchor_year_group as group_var,
        -- LABEL
        case 
            when datetime_diff(i.outtime,i.intime,hour)>72
            then 1 
            else 0 end as label
    from
        icu_last i
        left join `mimic-iv-ches.core.admissions` a on i.hadm_id = a.hadm_id
    where
        rn = 1
);
select * from `mimic-iv-ches.cohorts.mimic4ds_longlos`