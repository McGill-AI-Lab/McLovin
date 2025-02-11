from match_users import MatchMaker

if __name__ == "__main__":
    matcher = MatchMaker()
    matches = matcher.run_matching()

    # print out summary of matches with top 10
    print(f"\nMatching Summary:")
    print(f"Total matches generated: {len(matches)}")

    # top 10 by our very special scoring method
    print("\nTop 10 matches by score:")
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
    for user1_id, user2_id, score in sorted_matches[:10]:
        print(f"Match: {user1_id} - {user2_id}, Score: {score:.3f}")
