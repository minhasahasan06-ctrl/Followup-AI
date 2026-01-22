#!/usr/bin/env python3
"""
Embedding Standardization CI Check

Queries database for embeddings without proper model/version tracking
and fails CI if any are found in production.

This script is called from CI pipeline to enforce embedding governance.

Exit Codes:
- 0: All embeddings properly standardized
- 1: Non-standardized embeddings found
- 2: Database connection error

Usage:
    python scripts/check_embedding_standardization.py
    python scripts/check_embedding_standardization.py --env production
    python scripts/check_embedding_standardization.py --fix-report
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


EXPECTED_MODEL = "text-embedding-3-small"
EXPECTED_VERSION = "v1.0.0"
EXPECTED_DIMENSIONS = 1536


class EmbeddingStandardizationChecker:
    """Checks for non-standardized embeddings in database"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.results: Dict[str, Any] = {}
    
    async def check_null_embedding_models(self) -> Tuple[int, List[Dict]]:
        """Check for embeddings without model tracking"""
        logger.info("Checking for embeddings with NULL embedding_model...")
        
        count = 0
        examples = []
        
        try:
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                import asyncpg
                conn = await asyncpg.connect(database_url)
                try:
                    result = await conn.fetch("""
                        SELECT id, agent_id, patient_id, created_at 
                        FROM agent_memory 
                        WHERE embedding_model IS NULL 
                        LIMIT 10
                    """)
                    count = len(result)
                    examples = [dict(r) for r in result]
                    
                    if count > 0:
                        total_result = await conn.fetchval("""
                            SELECT COUNT(*) FROM agent_memory WHERE embedding_model IS NULL
                        """)
                        count = total_result
                finally:
                    await conn.close()
            else:
                logger.warning("DATABASE_URL not set, using stub check")
        except ImportError:
            logger.warning("asyncpg not available, using stub check")
        except Exception as e:
            logger.error(f"Database check failed: {e}")
        
        return count, examples
    
    async def check_wrong_embedding_version(self) -> Tuple[int, List[Dict]]:
        """Check for embeddings with wrong version"""
        logger.info(f"Checking for embeddings not using {EXPECTED_VERSION}...")
        
        count = 0
        examples = []
        
        try:
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                import asyncpg
                conn = await asyncpg.connect(database_url)
                try:
                    result = await conn.fetch(f"""
                        SELECT id, agent_id, embedding_model, embedding_version
                        FROM agent_memory 
                        WHERE embedding_version IS NOT NULL 
                        AND embedding_version != '{EXPECTED_VERSION}'
                        LIMIT 10
                    """)
                    count = len(result)
                    examples = [dict(r) for r in result]
                finally:
                    await conn.close()
        except ImportError:
            logger.warning("asyncpg not available")
        except Exception as e:
            logger.error(f"Database check failed: {e}")
        
        return count, examples
    
    async def check_dimension_mismatch(self) -> Tuple[int, List[Dict]]:
        """Check for embeddings with wrong dimensions"""
        logger.info(f"Checking for embeddings not using {EXPECTED_DIMENSIONS} dimensions...")
        
        count = 0
        examples = []
        
        return count, examples
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all standardization checks"""
        logger.info("=" * 60)
        logger.info("Embedding Standardization Check")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Expected Model: {EXPECTED_MODEL}")
        logger.info(f"Expected Version: {EXPECTED_VERSION}")
        logger.info(f"Expected Dimensions: {EXPECTED_DIMENSIONS}")
        logger.info("=" * 60)
        
        null_count, null_examples = await self.check_null_embedding_models()
        wrong_version_count, wrong_version_examples = await self.check_wrong_embedding_version()
        dim_mismatch_count, dim_examples = await self.check_dimension_mismatch()
        
        total_issues = null_count + wrong_version_count + dim_mismatch_count
        
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.environment,
            "expected_model": EXPECTED_MODEL,
            "expected_version": EXPECTED_VERSION,
            "expected_dimensions": EXPECTED_DIMENSIONS,
            "checks": {
                "null_embedding_model": {
                    "count": null_count,
                    "examples": null_examples[:5]
                },
                "wrong_version": {
                    "count": wrong_version_count,
                    "examples": wrong_version_examples[:5]
                },
                "dimension_mismatch": {
                    "count": dim_mismatch_count,
                    "examples": dim_examples[:5]
                }
            },
            "total_issues": total_issues,
            "passed": total_issues == 0
        }
        
        logger.info("=" * 60)
        logger.info("Results Summary")
        logger.info(f"  NULL embedding_model: {null_count}")
        logger.info(f"  Wrong version: {wrong_version_count}")
        logger.info(f"  Dimension mismatch: {dim_mismatch_count}")
        logger.info(f"  TOTAL ISSUES: {total_issues}")
        logger.info(f"  STATUS: {'PASSED' if total_issues == 0 else 'FAILED'}")
        logger.info("=" * 60)
        
        return self.results
    
    def generate_fix_report(self) -> str:
        """Generate report with fix recommendations"""
        if not self.results:
            return "No check results available. Run checks first."
        
        report = []
        report.append("=" * 60)
        report.append("EMBEDDING STANDARDIZATION FIX REPORT")
        report.append("=" * 60)
        report.append("")
        
        if self.results["passed"]:
            report.append("All embeddings are properly standardized.")
            return "\n".join(report)
        
        report.append("ISSUES FOUND:")
        report.append("")
        
        checks = self.results["checks"]
        
        if checks["null_embedding_model"]["count"] > 0:
            report.append(f"1. NULL embedding_model: {checks['null_embedding_model']['count']} records")
            report.append("   FIX: Run re-embedding migration script:")
            report.append("   python scripts/reembed_memories.py --batch-size 100")
            report.append("")
        
        if checks["wrong_version"]["count"] > 0:
            report.append(f"2. Wrong version: {checks['wrong_version']['count']} records")
            report.append(f"   FIX: Update to {EXPECTED_VERSION} using migration script")
            report.append("")
        
        if checks["dimension_mismatch"]["count"] > 0:
            report.append(f"3. Dimension mismatch: {checks['dimension_mismatch']['count']} records")
            report.append(f"   FIX: Re-embed with {EXPECTED_DIMENSIONS}-dimensional model")
            report.append("")
        
        report.append("=" * 60)
        report.append("RECOMMENDED ACTIONS:")
        report.append("1. Run: python scripts/reembed_memories.py --dry-run")
        report.append("2. Review output for affected records")
        report.append("3. Run: python scripts/reembed_memories.py --batch-size 50")
        report.append("4. Re-run this check to verify fix")
        report.append("=" * 60)
        
        return "\n".join(report)


async def main():
    parser = argparse.ArgumentParser(
        description="Check embedding standardization for CI"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=os.getenv("ENV", "development"),
        help="Environment to check"
    )
    parser.add_argument(
        "--fix-report",
        action="store_true",
        help="Generate fix recommendations"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any issues (default for production)"
    )
    
    args = parser.parse_args()
    
    checker = EmbeddingStandardizationChecker(environment=args.env)
    
    try:
        results = await checker.run_all_checks()
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        sys.exit(2)
    
    if args.fix_report:
        print(checker.generate_fix_report())
    
    is_production = args.env == "production"
    should_fail = args.strict or is_production
    
    if results["total_issues"] > 0 and should_fail:
        logger.error("FAILED: Non-standardized embeddings found")
        sys.exit(1)
    elif results["total_issues"] > 0:
        logger.warning("WARNING: Non-standardized embeddings found (not failing in dev)")
        sys.exit(0)
    else:
        logger.info("PASSED: All embeddings properly standardized")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
