from match_users import MatchMaker
import logging

if __name__ == "__main__":
    matcher = MatchMaker()
    matches = matcher.run_matching()

    # Print detailed summary
    print("\nMatching Results:")
    if matches:
        print(f"\nTotal matches made: {len(matches)}")
        print("\nTop matches by score:")
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
        for user1_id, user2_id, score in sorted_matches[:10]:
            print(f"Match: {user1_id} - {user2_id} (Score: {score:.3f})")
    else:
        print("No matches were generated")
