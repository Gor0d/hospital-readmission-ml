"""
Testes unitários para utils/preprocessing.py
"""

import os
import sys
import numpy as np
import joblib
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.preprocessing import build_feature_vector, classify_risk, RISK_LOW, RISK_MODERATE

MODEL_DIR = os.path.join(ROOT, "model")


@pytest.fixture(scope="module")
def encoders():
    return joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))


class TestBuildFeatureVector:
    def test_output_shape(self, sample_patient, encoders):
        X = build_feature_vector(sample_patient, encoders)
        assert X.shape == (1, 19), f"Shape esperado (1, 19), obtido {X.shape}"

    def test_output_dtype(self, sample_patient, encoders):
        X = build_feature_vector(sample_patient, encoders)
        assert X.dtype == np.float64

    def test_no_nan_for_valid_input(self, sample_patient, encoders):
        X = build_feature_vector(sample_patient, encoders)
        assert not np.isnan(X).any(), "Feature vector não deve conter NaN para input válido"

    def test_none_string_handled(self, sample_patient, encoders):
        """'None' como string deve ser tratado sem erro (mapeia para NaN → encoder)."""
        data = sample_patient.copy()
        data["hba1c_result"] = "None"
        data["glucose_serum_test"] = "None"
        # Não deve levantar exceção
        X = build_feature_vector(data, encoders)
        assert X.shape == (1, 19)

    def test_invalid_category_raises_value_error(self, sample_patient, encoders):
        data = sample_patient.copy()
        data["gender"] = "Outro"
        with pytest.raises(ValueError, match="Valor inválido para 'gender'"):
            build_feature_vector(data, encoders)

    def test_risk_score_calculation(self, encoders):
        """Verifica que risk_score = inpatient*2 + emergency + indicadores clínicos."""
        data = {
            "age_numeric": 50, "gender": "Male", "diag_primary": "Diabetes",
            "time_in_hospital": 3, "num_medications": 10, "num_procedures": 1,
            "num_diagnoses": 4, "num_lab_procedures": 30,
            "number_outpatient": 0, "number_emergency": 2, "number_inpatient": 1,
            "hba1c_result": ">8", "glucose_serum_test": ">200",
            "insulin": "Up", "change_medications": "Ch", "diabetes_medication": "Yes",
        }
        X = build_feature_vector(data, encoders)
        # risk_score = 1*2 + 2 + 1 + 1 = 6 (índice 16)
        assert X[0, 16] == 6.0

    def test_medication_complexity(self, encoders):
        """medication_complexity = num_medications * num_diagnoses (índice 17)."""
        data = {
            "age_numeric": 60, "gender": "Female", "diag_primary": "Circulatory",
            "time_in_hospital": 4, "num_medications": 5, "num_procedures": 0,
            "num_diagnoses": 3, "num_lab_procedures": 20,
            "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
            "hba1c_result": "Normal", "glucose_serum_test": "Normal",
            "insulin": "No", "change_medications": "No", "diabetes_medication": "No",
        }
        X = build_feature_vector(data, encoders)
        assert X[0, 17] == 15.0  # 5 * 3

    def test_hospital_utilization(self, encoders):
        """hospital_utilization = outpatient + emergency + inpatient (índice 18)."""
        data = {
            "age_numeric": 70, "gender": "Male", "diag_primary": "Respiratory",
            "time_in_hospital": 7, "num_medications": 12, "num_procedures": 2,
            "num_diagnoses": 6, "num_lab_procedures": 50,
            "number_outpatient": 1, "number_emergency": 2, "number_inpatient": 3,
            "hba1c_result": "None", "glucose_serum_test": "None",
            "insulin": "Steady", "change_medications": "No", "diabetes_medication": "Yes",
        }
        X = build_feature_vector(data, encoders)
        assert X[0, 18] == 6.0  # 1 + 2 + 3


class TestClassifyRisk:
    def test_low_risk(self):
        risk, rec = classify_risk(RISK_LOW - 0.01)
        assert risk == "Baixo"
        assert "padrão" in rec.lower()

    def test_moderate_risk(self):
        risk, rec = classify_risk((RISK_LOW + RISK_MODERATE) / 2)
        assert risk == "Moderado"
        assert "ambulatorial" in rec.lower()

    def test_high_risk(self):
        risk, rec = classify_risk(RISK_MODERATE + 0.01)
        assert risk == "Alto"
        assert "intensivo" in rec.lower()

    def test_boundary_low_moderate(self):
        """Exatamente no limiar baixo/moderado."""
        risk, _ = classify_risk(RISK_LOW)
        assert risk == "Moderado"

    def test_boundary_moderate_high(self):
        """Exatamente no limiar moderado/alto."""
        risk, _ = classify_risk(RISK_MODERATE)
        assert risk == "Alto"

    def test_probability_zero(self):
        risk, _ = classify_risk(0.0)
        assert risk == "Baixo"

    def test_probability_one(self):
        risk, _ = classify_risk(1.0)
        assert risk == "Alto"
