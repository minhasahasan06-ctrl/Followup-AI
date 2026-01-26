"""
Data Warehouse Aggregation Jobs
Production-grade ETL jobs for epidemiology data warehouse.
Runs nightly aggregations with k-anonymity and privacy protections.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func, text, and_, or_

from app.database import SessionLocal
from app.models.epidemiology_models import (
    EpidemiologyCase,
    InfectiousEvent,
    OccupationalIncident,
    GenomicMarkerStat,
    DrugSafetySignal,
    VaccineCoverageRecord,
    VaccineEffectivenessRecord,
    DailySurveillanceAggregate,
    WeeklyIncidenceSummary,
    OccupationalCohort,
    SurveillanceLocation,
)
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)

K_ANONYMITY_THRESHOLD = 5


class WarehouseAggregationService:
    """
    Data warehouse aggregation service.
    Runs ETL jobs to populate fact tables from transactional data.
    """

    def __init__(self, db: Session):
        self.db = db

    def _log_etl_access(self, job_type: str, details: Optional[Dict] = None) -> str:
        """Log ETL job execution for audit"""
        return HIPAAAuditLogger.log_phi_access(
            actor_id="system",
            actor_role="etl_job",
            patient_id="aggregate",
            action=f"etl_{job_type}",
            phi_categories=["epidemiology", "aggregate_data"],
            resource_type="data_warehouse",
            access_scope="system",
            access_reason="warehouse_aggregation",
            consent_verified=True,
            additional_context=details,
        )

    def aggregate_daily_surveillance(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Aggregate daily surveillance data from case-level records.
        Applies k-anonymity suppression.
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        self._log_etl_access("daily_surveillance", {"target_date": target_date.isoformat()})

        aggregation_query = (
            self.db.query(
                EpidemiologyCase.condition_code,
                EpidemiologyCase.condition_name,
                EpidemiologyCase.location_id,
                EpidemiologyCase.geo_region,
                func.count(EpidemiologyCase.id).label("case_count"),
                func.sum(
                    func.cast(
                        EpidemiologyCase.outcome == "death",
                        type_=self.db.bind.dialect.type_descriptor(__import__('sqlalchemy').Integer)
                    )
                ).label("death_count"),
                func.sum(
                    func.cast(
                        EpidemiologyCase.hospitalized == True,
                        type_=self.db.bind.dialect.type_descriptor(__import__('sqlalchemy').Integer)
                    )
                ).label("hospitalization_count"),
                func.sum(
                    func.cast(
                        EpidemiologyCase.icu_admission == True,
                        type_=self.db.bind.dialect.type_descriptor(__import__('sqlalchemy').Integer)
                    )
                ).label("icu_count"),
            )
            .filter(EpidemiologyCase.onset_date == target_date)
            .group_by(
                EpidemiologyCase.condition_code,
                EpidemiologyCase.condition_name,
                EpidemiologyCase.location_id,
                EpidemiologyCase.geo_region,
            )
        )

        results = aggregation_query.all()
        records_created = 0
        records_suppressed = 0

        for row in results:
            case_count = int(row.case_count or 0)
            suppressed = case_count < K_ANONYMITY_THRESHOLD

            if suppressed:
                records_suppressed += 1
                case_count = 0

            existing = self.db.query(DailySurveillanceAggregate).filter(
                and_(
                    DailySurveillanceAggregate.report_date == target_date,
                    DailySurveillanceAggregate.condition_code == row.condition_code,
                    DailySurveillanceAggregate.location_id == row.location_id,
                )
            ).first()

            if existing:
                existing.case_count = case_count
                existing.death_count = int(row.death_count or 0) if not suppressed else 0
                existing.hospitalization_count = int(row.hospitalization_count or 0) if not suppressed else 0
                existing.icu_count = int(row.icu_count or 0) if not suppressed else 0
                existing.suppressed = suppressed
                existing.suppression_reason = "k_anonymity" if suppressed else None
            else:
                cumulative = self._calculate_cumulative_cases(
                    row.condition_code, row.location_id, target_date
                )

                new_record = DailySurveillanceAggregate(
                    report_date=target_date,
                    condition_code=row.condition_code,
                    condition_name=row.condition_name,
                    location_id=row.location_id,
                    geo_region=row.geo_region,
                    case_count=case_count,
                    death_count=int(row.death_count or 0) if not suppressed else 0,
                    hospitalization_count=int(row.hospitalization_count or 0) if not suppressed else 0,
                    icu_count=int(row.icu_count or 0) if not suppressed else 0,
                    cumulative_cases=cumulative + case_count,
                    cumulative_deaths=0,
                    suppressed=suppressed,
                    suppression_reason="k_anonymity" if suppressed else None,
                )
                self.db.add(new_record)
                records_created += 1

        self.db.commit()

        return {
            "job": "daily_surveillance_aggregation",
            "target_date": target_date.isoformat(),
            "records_created": records_created,
            "records_suppressed": records_suppressed,
            "status": "completed",
        }

    def _calculate_cumulative_cases(
        self, condition_code: str, location_id: Optional[str], as_of_date: date
    ) -> int:
        """Calculate cumulative case count up to a date"""
        query = self.db.query(func.sum(DailySurveillanceAggregate.case_count)).filter(
            and_(
                DailySurveillanceAggregate.condition_code == condition_code,
                DailySurveillanceAggregate.report_date < as_of_date,
            )
        )
        if location_id:
            query = query.filter(DailySurveillanceAggregate.location_id == location_id)

        result = query.scalar()
        return int(result or 0)

    def aggregate_weekly_incidence(self, week_end: Optional[date] = None) -> Dict[str, Any]:
        """
        Aggregate weekly incidence summaries from daily data.
        Calculates incidence rates and trends.
        """
        if week_end is None:
            today = date.today()
            days_since_sunday = (today.weekday() + 1) % 7
            week_end = today - timedelta(days=days_since_sunday)

        week_start = week_end - timedelta(days=6)
        prev_week_start = week_start - timedelta(days=7)
        prev_week_end = week_start - timedelta(days=1)

        self._log_etl_access("weekly_incidence", {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
        })

        epi_week = week_start.isocalendar()[1]
        epi_year = week_start.isocalendar()[0]

        current_week_query = (
            self.db.query(
                DailySurveillanceAggregate.condition_code,
                DailySurveillanceAggregate.condition_name,
                DailySurveillanceAggregate.location_id,
                DailySurveillanceAggregate.geo_region,
                func.sum(DailySurveillanceAggregate.case_count).label("case_count"),
                func.sum(DailySurveillanceAggregate.death_count).label("death_count"),
            )
            .filter(
                and_(
                    DailySurveillanceAggregate.report_date >= week_start,
                    DailySurveillanceAggregate.report_date <= week_end,
                    DailySurveillanceAggregate.suppressed == False,
                )
            )
            .group_by(
                DailySurveillanceAggregate.condition_code,
                DailySurveillanceAggregate.condition_name,
                DailySurveillanceAggregate.location_id,
                DailySurveillanceAggregate.geo_region,
            )
        )

        results = current_week_query.all()
        records_created = 0

        for row in results:
            case_count = int(row.case_count or 0)
            if case_count < K_ANONYMITY_THRESHOLD:
                continue

            prev_week_cases = (
                self.db.query(func.sum(DailySurveillanceAggregate.case_count))
                .filter(
                    and_(
                        DailySurveillanceAggregate.condition_code == row.condition_code,
                        DailySurveillanceAggregate.location_id == row.location_id,
                        DailySurveillanceAggregate.report_date >= prev_week_start,
                        DailySurveillanceAggregate.report_date <= prev_week_end,
                    )
                )
                .scalar()
            )
            prev_cases = int(prev_week_cases or 0)

            if prev_cases > 0:
                percent_change = ((case_count - prev_cases) / prev_cases) * 100
            else:
                percent_change = 100 if case_count > 0 else 0

            if percent_change > 10:
                trend_direction = "increasing"
            elif percent_change < -10:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            location = self.db.query(SurveillanceLocation).filter(
                SurveillanceLocation.id == row.location_id
            ).first()
            population = location.population if location else 100000
            incidence_per_100k = (case_count / population) * 100000 if population else 0

            existing = self.db.query(WeeklyIncidenceSummary).filter(
                and_(
                    WeeklyIncidenceSummary.week_start == week_start,
                    WeeklyIncidenceSummary.condition_code == row.condition_code,
                    WeeklyIncidenceSummary.location_id == row.location_id,
                )
            ).first()

            if existing:
                existing.case_count = case_count
                existing.death_count = int(row.death_count or 0)
                existing.incidence_rate_per_100k = round(incidence_per_100k, 2)
                existing.percent_change_prev_week = round(percent_change, 1)
                existing.trend_direction = trend_direction
            else:
                new_summary = WeeklyIncidenceSummary(
                    week_start=week_start,
                    week_end=week_end,
                    epi_week=epi_week,
                    epi_year=epi_year,
                    condition_code=row.condition_code,
                    condition_name=row.condition_name,
                    location_id=row.location_id,
                    geo_region=row.geo_region,
                    case_count=case_count,
                    death_count=int(row.death_count or 0),
                    incidence_rate_per_100k=round(incidence_per_100k, 2),
                    population_denominator=population,
                    percent_change_prev_week=round(percent_change, 1),
                    trend_direction=trend_direction,
                    suppressed=False,
                )
                self.db.add(new_summary)
                records_created += 1

        self.db.commit()

        return {
            "job": "weekly_incidence_aggregation",
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "epi_week": epi_week,
            "epi_year": epi_year,
            "records_created": records_created,
            "status": "completed",
        }

    def calculate_r_values(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate R0/Rt values for infectious conditions.
        Uses simplified serial interval estimation.
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        self._log_etl_access("r_value_calculation", {"target_date": target_date.isoformat()})

        window_size = 7
        window_start = target_date - timedelta(days=window_size)
        prev_window_start = window_start - timedelta(days=window_size)
        prev_window_end = window_start - timedelta(days=1)

        conditions = (
            self.db.query(DailySurveillanceAggregate.condition_code)
            .filter(DailySurveillanceAggregate.condition_code.like('%COVID%') |
                    DailySurveillanceAggregate.condition_code.like('%INFLUENZA%') |
                    DailySurveillanceAggregate.condition_code.like('%RSV%'))
            .distinct()
            .all()
        )

        updates = 0
        for (condition_code,) in conditions:
            current_cases = (
                self.db.query(func.sum(DailySurveillanceAggregate.case_count))
                .filter(
                    and_(
                        DailySurveillanceAggregate.condition_code == condition_code,
                        DailySurveillanceAggregate.report_date >= window_start,
                        DailySurveillanceAggregate.report_date <= target_date,
                    )
                )
                .scalar()
            ) or 0

            prev_cases = (
                self.db.query(func.sum(DailySurveillanceAggregate.case_count))
                .filter(
                    and_(
                        DailySurveillanceAggregate.condition_code == condition_code,
                        DailySurveillanceAggregate.report_date >= prev_window_start,
                        DailySurveillanceAggregate.report_date <= prev_window_end,
                    )
                )
                .scalar()
            ) or 0

            if prev_cases > 0:
                r_value = current_cases / prev_cases
            else:
                r_value = 1.0 if current_cases > 0 else 0.0

            se = 0.1 * r_value if r_value > 0 else 0.1
            r_lower = max(0, r_value - 1.96 * se)
            r_upper = r_value + 1.96 * se

            record = self.db.query(DailySurveillanceAggregate).filter(
                and_(
                    DailySurveillanceAggregate.condition_code == condition_code,
                    DailySurveillanceAggregate.report_date == target_date,
                )
            ).first()

            if record:
                record.r_value = round(r_value, 2)
                record.r_lower = round(r_lower, 2)
                record.r_upper = round(r_upper, 2)
                updates += 1

        self.db.commit()

        return {
            "job": "r_value_calculation",
            "target_date": target_date.isoformat(),
            "conditions_processed": len(conditions),
            "records_updated": updates,
            "status": "completed",
        }

    def update_occupational_cohorts(self) -> Dict[str, Any]:
        """
        Update occupational cohort statistics from incident data.
        """
        self._log_etl_access("occupational_cohort_update")

        industry_stats = (
            self.db.query(
                OccupationalIncident.industry_code,
                OccupationalIncident.industry_name,
                OccupationalIncident.hazard_code,
                OccupationalIncident.hazard_name,
                func.count(func.distinct(OccupationalIncident.patient_id)).label("n_workers"),
                func.avg(OccupationalIncident.cumulative_exposure_years).label("mean_exposure"),
            )
            .group_by(
                OccupationalIncident.industry_code,
                OccupationalIncident.industry_name,
                OccupationalIncident.hazard_code,
                OccupationalIncident.hazard_name,
            )
        ).all()

        cohorts_updated = 0
        for row in industry_stats:
            n_workers = int(row.n_workers or 0)
            if n_workers < K_ANONYMITY_THRESHOLD:
                continue

            existing = self.db.query(OccupationalCohort).filter(
                and_(
                    OccupationalCohort.industry_code == row.industry_code,
                    OccupationalCohort.hazard_code == row.hazard_code,
                )
            ).first()

            if existing:
                existing.n_workers = n_workers
                existing.mean_exposure_years = float(row.mean_exposure or 0)
            else:
                new_cohort = OccupationalCohort(
                    cohort_name=f"{row.industry_name} - {row.hazard_name}",
                    industry_code=row.industry_code,
                    industry_name=row.industry_name,
                    hazard_code=row.hazard_code,
                    hazard_name=row.hazard_name,
                    n_workers=n_workers,
                    mean_exposure_years=float(row.mean_exposure or 0),
                )
                self.db.add(new_cohort)

            cohorts_updated += 1

        self.db.commit()

        return {
            "job": "occupational_cohort_update",
            "cohorts_updated": cohorts_updated,
            "status": "completed",
        }

    def run_all_aggregations(self) -> Dict[str, Any]:
        """
        Run all warehouse aggregation jobs.
        Called by nightly scheduler.
        """
        logger.info("🏭 Starting data warehouse aggregation jobs...")

        results = {}

        try:
            results["daily_surveillance"] = self.aggregate_daily_surveillance()
            logger.info(f"✅ Daily surveillance: {results['daily_surveillance']['records_created']} records")
        except Exception as e:
            logger.error(f"❌ Daily surveillance failed: {e}")
            results["daily_surveillance"] = {"status": "failed", "error": str(e)}

        try:
            results["weekly_incidence"] = self.aggregate_weekly_incidence()
            logger.info(f"✅ Weekly incidence: {results['weekly_incidence']['records_created']} records")
        except Exception as e:
            logger.error(f"❌ Weekly incidence failed: {e}")
            results["weekly_incidence"] = {"status": "failed", "error": str(e)}

        try:
            results["r_values"] = self.calculate_r_values()
            logger.info(f"✅ R-value calculation: {results['r_values']['records_updated']} updates")
        except Exception as e:
            logger.error(f"❌ R-value calculation failed: {e}")
            results["r_values"] = {"status": "failed", "error": str(e)}

        try:
            results["occupational_cohorts"] = self.update_occupational_cohorts()
            logger.info(f"✅ Occupational cohorts: {results['occupational_cohorts']['cohorts_updated']} cohorts")
        except Exception as e:
            logger.error(f"❌ Occupational cohorts failed: {e}")
            results["occupational_cohorts"] = {"status": "failed", "error": str(e)}

        logger.info("🏭 Data warehouse aggregation jobs completed")

        return {
            "job": "all_aggregations",
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }


async def run_nightly_warehouse_aggregation():
    """
    Async entry point for nightly warehouse aggregation.
    Called by scheduler.
    """
    db = None
    try:
        db = SessionLocal()
        service = WarehouseAggregationService(db)
        result = service.run_all_aggregations()
        logger.info(f"🏭 Nightly aggregation completed: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Nightly aggregation failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        if db:
            db.close()
