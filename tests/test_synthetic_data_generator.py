from src.utils.synthetic_data_generator import generate_age, generate_user_id_str


def test_generate_user_id_str():
    user_ids = generate_user_id_str(5, seed=0)
    assert len(user_ids) == 5
    assert len(set(user_ids)) == 5  # Check if all user ids are unique
    assert all(len(user_id) == 10 for user_id in user_ids)


def test_generate_age():
    ages = generate_age(5)
    assert len(ages) == 5
    assert all(18 <= age < 70 for age in ages)
